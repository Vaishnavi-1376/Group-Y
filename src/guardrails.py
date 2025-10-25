"""
Safety Guardrails Module

Provides 3-layer safety validation:
 1. Hallucination detection
 2. Dosage validation
 3. Drug interaction checking

Designed to be imported by other members (Member 2 / Member 4)
"""
from typing import Dict, List, Any, Tuple
import os
import re

from src.config import DATA_PATH
from src.utils import log_event, print_header, load_json, save_json, clean_text
from src.llm_adapter import embed_text, generate_text


class SafetyGuardrails:
    def __init__(self, retriever=None):
        self.retriever = retriever
        self.dangerous_pairs = self._load_dangerous_pairs()
        # lightweight max dose map for MVP (mg/day unless noted)
        self._max_dose_map = {
            'aspirin': 4000,   # mg/day
            'metformin': 2550, # mg/day (typical prescription limits)
            'ibuprofen': 3200, # mg/day
        }
        log_event("✓ Safety guardrails initialized")

    def _load_dangerous_pairs(self) -> Dict[str, Any]:
        pairs_file = os.path.join(DATA_PATH, 'dangerous_drug_pairs.json')
        if os.path.exists(pairs_file):
            try:
                return load_json(pairs_file)
            except Exception as e:
                log_event(f"Failed to load dangerous pairs: {e}", "ERROR")
                return {"known_dangerous_pairs": [], "known_safe_pairs": []}
        else:
            log_event("dangerous_drug_pairs.json not found", "WARN")
            return {"known_dangerous_pairs": [], "known_safe_pairs": []}

    # ==================== CHECK 1: HALLUCINATION DETECTION ====================
    def hallucination_check(self, response_data: Dict, retrieved_chunks: List[Dict]) -> Dict:
        """
        CHECK 1: Detect if response is hallucinated

        Factors: confidence score (40%), citation presence (30%), retrieved chunks count (30%)
        Simple rule-based thresholds as defined in the HLD.
        """
        print_header("CHECK 1: HALLUCINATION DETECTION")

        confidence_score = float(response_data.get('confidence', 0.0) or 0.0)
        citations = response_data.get('citations', []) or []
        has_citations = len(citations) > 0
        chunk_count = len(retrieved_chunks or [])

        log_event(f"  Confidence score: {confidence_score:.3f}")
        log_event(f"  Citations count: {len(citations)}")
        log_event(f"  Retrieved chunk count: {chunk_count}")

        # Decision logic per HLD
        if confidence_score < 0.40:
            is_hallucinated = True
            severity = "HIGH"
            reasoning = f"CRITICAL: Confidence too low ({confidence_score:.3f}). Not grounded in FDA labels."
        elif confidence_score < 0.60:
            is_hallucinated = True
            severity = "MEDIUM"
            reasoning = f"LOW CONFIDENCE ({confidence_score:.3f}). May not be fully grounded."
        elif (not has_citations) and (chunk_count < 5):
            is_hallucinated = True
            severity = "MEDIUM"
            reasoning = "No citations and insufficient retrieved chunks (need >=5). Response may not be grounded."
        else:
            is_hallucinated = False
            severity = "NONE"
            reasoning = f"GOOD: Confidence {confidence_score:.3f}. Response appears grounded."

        recommendation = "REJECT" if severity == "HIGH" and is_hallucinated else ("WARN" if is_hallucinated else "ACCEPT")

        log_event(f"  Status: {'HALLUCINATED' if is_hallucinated else 'GROUNDED'}")
        log_event(f"  Severity: {severity}")
        log_event(f"  Recommendation: {recommendation}")

        return {
            "hallucination_detected": is_hallucinated,
            "confidence_score": confidence_score,
            "severity": severity,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "citations_count": len(citations),
            "chunks_count": chunk_count,
        }

    # ==================== CHECK 2: DOSAGE VALIDATION ====================
    def _parse_doses_from_text(self, text: str) -> List[Tuple[float, str]]:
        """Return list of (value, unit) from text. Units normalized to mg when possible."""
        if not text:
            return []
        # capture numeric and unit
        pattern = r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|ug|ml|mL)"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        parsed = []
        for val, unit in matches:
            try:
                v = float(val)
            except Exception:
                continue
            u = unit.lower()
            # normalize to mg where straightforward
            if u == 'g':
                v_mg = v * 1000
                parsed.append((v_mg, 'mg'))
            elif u in ('mcg', 'µg', 'ug'):
                v_mg = v / 1000.0
                parsed.append((v_mg, 'mg'))
            elif u in ('ml', 'mL'):
                parsed.append((v, 'ml'))
            else:
                parsed.append((v, 'mg'))
        return parsed

    def validate_dosage(self, drug_name: str, dosage_text: str) -> Dict:
        """
        CHECK 2: Validate that recommended dosage is safe.

        For MVP we extract numeric doses and compare to a small in-file max-dose map.
        If retriever integration exists, this function can be extended to fetch FDA labels.
        """
        print_header("CHECK 2: DOSAGE VALIDATION")
        drug_key = (drug_name or '').lower().strip()
        log_event(f"  Drug: {drug_key}")
        log_event(f"  Dosage text (snippet): {dosage_text[:120]}")

        extracted = self._parse_doses_from_text(dosage_text or '')
        # parse frequency words to estimate times per day
        freq_text = (dosage_text or '').lower()
        times_per_day = 1
        # common frequency keywords (check specific before generic 'daily')
        if re.search(r'\bthree times daily\b|\btid\b|\b3 times? daily\b', freq_text):
            times_per_day = 3
        elif re.search(r'\btwice daily\b|\btwice a day\b|\bbid\b|\b2 times? daily\b', freq_text):
            times_per_day = 2
        elif re.search(r'\bonce daily\b|\bonce a day\b|\bdaily\b', freq_text):
            times_per_day = 1
        else:
            # look for patterns like 'every 8 hours' or 'q8h'
            m = re.search(r'every\s+(\d+)\s*hours?', freq_text)
            if m:
                hours = int(m.group(1))
                if hours > 0:
                    times_per_day = max(1, int(round(24 / hours)))
            else:
                m2 = re.search(r'q(\d+)h', freq_text)
                if m2:
                    hours = int(m2.group(1))
                    if hours > 0:
                        times_per_day = max(1, int(round(24 / hours)))
        if not extracted:
            log_event("  No explicit dosage found in response text")
            return {
                "dosage_valid": True,
                "max_dose": "Unknown",
                "recommended_dose": None,
                "warning": None,
                "severity": "NONE",
            }

        # Sum mg entries (ignore ml for now) and multiply by frequency
        per_dose_mg = sum(v for v, u in extracted if u == 'mg')
        total_mg = per_dose_mg * times_per_day

        rec_str = ", ".join([f"{v}{u}" for v, u in extracted])
        if times_per_day != 1:
            rec_str = f"{rec_str} x {times_per_day}/day"

        # Attempt to fetch authoritative max dose from retriever (if configured)
        max_allowed = None
        if self.retriever and hasattr(self.retriever, 'get_label_text'):
            try:
                # Prefer embedding-based grounding: pick top-k relevant chunks via embeddings
                try:
                    chunks = []
                    if hasattr(self.retriever, 'retrieve_chunks'):
                        r = self.retriever.retrieve_chunks(drug_key, n_results=20)
                        chunks = r.get('chunks', [])
                    else:
                        # fallback to single label text
                        chunks = [{'text': self.retriever.get_label_text(drug_key)}]

                    if chunks:
                        # compute embedding for the dosage_text and each chunk, then score
                        q_vec = embed_text(dosage_text or drug_key)
                        scored = []
                        for c in chunks:
                            txt = c.get('text', '')
                            try:
                                v = embed_text(txt)
                                # cosine similarity
                                num = sum(a*b for a, b in zip(q_vec, v))
                                den_a = sum(a*a for a in q_vec) ** 0.5
                                den_b = sum(b*b for b in v) ** 0.5
                                sim = num / (den_a * den_b) if den_a and den_b else 0.0
                            except Exception:
                                sim = 0.0
                            scored.append((sim, txt))

                        scored = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
                        combined = "\n".join(t for s, t in scored)
                    else:
                        combined = self.retriever.get_label_text(drug_key) or ''

                    label_text = combined
                except Exception as e:
                    log_event(f"  Embedding grounding failed: {e}", "WARN")
                    label_text = self.retriever.get_label_text(drug_key)

                if label_text:
                    # Try to find patterns like 'maximum recommended dose is 4,000 mg' or 'max dose: 4000 mg/day' or 'do not exceed 4 g in 24 hours'
                    patterns = [
                        r"max(?:imum)?(?: recommended)?(?: total)?(?: dose)?[:\s]*?(?:is\s*)?(\d{1,6}(?:,\d{3})?)\s*(mg|g)",
                        r"do not exceed\s*(\d{1,6}(?:,\d{3})?)\s*(mg|g)\s*(?:in\s*24\s*hours)?",
                        r"(\d{1,6}(?:,\d{3})?)\s*[-–]\s*(\d{1,6}(?:,\d{3})?)\s*(mg|g)",
                        r"(\d{1,6}(?:,\d{3})?)\s*(mg|g)\s*/?\s*(?:24\s*hours|day)"
                    ]
                    found = None
                    for pat in patterns:
                        m = re.search(pat, label_text, flags=re.IGNORECASE)
                        if m:
                            found = m
                            break
                    if found:
                        # support range capture
                        groups = found.groups()
                        # find numeric group and unit
                        nums = [g for g in groups if g and re.search(r"\d", str(g))]
                        unit = None
                        val = None
                        if len(nums) >= 2 and re.search(r"mg|g", ' '.join(groups), flags=re.IGNORECASE):
                            # range like '500-1000 mg' -> take upper bound
                            try:
                                val = float(str(nums[-1]).replace(',', ''))
                            except Exception:
                                val = None
                        else:
                            try:
                                val = float(str(nums[0]).replace(',', ''))
                            except Exception:
                                val = None
                        # unit detection
                        if re.search(r"g", label_text, flags=re.IGNORECASE):
                            unit = 'g'
                        elif re.search(r"mg", label_text, flags=re.IGNORECASE):
                            unit = 'mg'

                        if val is not None and unit:
                            if unit == 'g':
                                val = val * 1000
                            max_allowed = int(val)
                            log_event(f"  Extracted max dose from label for {drug_key}: {max_allowed} mg/day")
            except Exception as e:
                log_event(f"  Error extracting max dose from retriever: {e}", "WARN")

        if max_allowed is None:
            max_allowed = self._max_dose_map.get(drug_key)
        if max_allowed is None:
            log_event(f"  No max-dose info for {drug_key}; cannot fully validate")
            return {
                "dosage_valid": True,
                "max_dose": "Unknown",
                "recommended_dose": rec_str,
                "warning": None,
                "severity": "NONE",
            }

        log_event(f"  Recommended total (mg): {total_mg}")
        log_event(f"  Max allowed (mg/day): {max_allowed}")

        if total_mg > max_allowed:
            warning = f"Recommended total {total_mg} mg exceeds max {max_allowed} mg/day"
            log_event(f"  {warning}", "WARN")
            return {
                "dosage_valid": False,
                "max_dose": f"{max_allowed} mg/day",
                "recommended_dose": rec_str,
                "warning": warning,
                "severity": "HIGH",
            }

        return {
            "dosage_valid": True,
            "max_dose": f"{max_allowed} mg/day",
            "recommended_dose": rec_str,
            "warning": None,
            "severity": "NONE",
        }

    # ==================== CHECK 3: INTERACTION CHECKING ====================
    def check_interaction(self, drug1: str, drug2: str) -> Dict:
        print_header("CHECK 3: INTERACTION CHECKING")
        d1 = (drug1 or '').lower().strip()
        d2 = (drug2 or '').lower().strip()
        log_event(f"  Drug 1: {drug1} -> {d1}")
        log_event(f"  Drug 2: {drug2} -> {d2}")

        for pair in self.dangerous_pairs.get('known_dangerous_pairs', []):
            a = pair.get('drug1', '').lower().strip()
            b = pair.get('drug2', '').lower().strip()
            if (d1 == a and d2 == b) or (d1 == b and d2 == a):
                log_event(f"  🚫 DANGEROUS combination detected: {a} + {b}")
                return {
                    "interaction_detected": True,
                    "severity": pair.get('severity', 'HIGH'),
                    "warning": pair.get('reason'),
                    "recommendation": "REJECT - Do not recommend together",
                    "evidence": pair.get('reason')
                }

        log_event("  ✓ No dangerous interaction detected")
        return {
            "interaction_detected": False,
            "severity": "NONE",
            "warning": None,
            "recommendation": "ACCEPT",
            "evidence": None,
        }

    # ==================== INTEGRATION: RUN ALL CHECKS ====================
    def run_all_safety_checks(self, response_data: Dict, query: str, retrieved_chunks: List[Dict], generate_suggestion: bool = False) -> Dict:
        print_header("RUNNING ALL SAFETY CHECKS")
        log_event(f"Query snippet: {str(query)[:120]}")

        halluc = self.hallucination_check(response_data, retrieved_chunks)
        dosage = self.validate_dosage(self._find_drug_in_text(query) or 'drug', response_data.get('response', ''))

        # find drug mentions (from known pairs + safe pairs)
        known = set()
        for p in self.dangerous_pairs.get('known_dangerous_pairs', []):
            known.add(p.get('drug1', '').lower())
            known.add(p.get('drug2', '').lower())
        for p in self.dangerous_pairs.get('known_safe_pairs', []):
            known.add(p.get('drug1', '').lower())
            known.add(p.get('drug2', '').lower())

        found = []
        text_to_search = (query or '') + '\n' + response_data.get('response', '')
        text_to_search = clean_text(text_to_search).lower()
        for drug in known:
            if not drug:
                continue
            # simple word boundary match
            if re.search(rf'\b{re.escape(drug)}\b', text_to_search):
                found.append(drug)

        interaction_result = {"interaction_detected": False, "severity": "NONE"}
        # check pairwise
        if len(found) >= 2:
            # check all pairs, if any HIGH then raise
            for i in range(len(found)):
                for j in range(i+1, len(found)):
                    res = self.check_interaction(found[i], found[j])
                    if res.get('interaction_detected'):
                        interaction_result = res
                        break
                if interaction_result.get('interaction_detected'):
                    break

        # Decision logic
    if halluc.get('severity') == 'HIGH':
            overall = 'CRITICAL'
            final = '🚫 REJECT: Response likely hallucinated'
            reason = halluc.get('reasoning')
        elif dosage.get('severity') == 'HIGH':
            overall = 'CRITICAL'
            final = '🚫 REJECT: Dosage exceeds known maximum'
            reason = dosage.get('warning')
        elif interaction_result.get('severity') == 'HIGH':
            overall = 'CRITICAL'
            final = '🚫 REJECT: Dangerous drug combination'
            reason = interaction_result.get('warning')
        elif halluc.get('severity') == 'MEDIUM':
            overall = 'WARNING'
            final = '⚠ WARNING: Low confidence response'
            reason = halluc.get('reasoning')
        else:
            overall = 'SAFE'
            final = '✅ ACCEPT: Response appears safe'
            reason = 'All safety checks passed'

        log_event(f"Overall Status: {overall}")
        log_event(f"Final recommendation: {final}")

        return {
            'hallucination_check': halluc,
            'dosage_validation': dosage,
        suggestion = None
        if generate_suggestion and overall != 'SAFE':
            try:
                prompt = f"The following model response was flagged as {overall}:\n\nQuery: {query}\n\nResponse: {response_data.get('response', '')}\n\nPlease produce a safe, concise, and authoritative rephrasing that avoids dosing advice or warns the user and cites that they should consult official FDA labels and a clinician." 
                suggestion = generate_text(prompt, max_tokens=200)
            except Exception as e:
                log_event(f"Failed to generate suggestion: {e}", 'WARN')
            'overall_safety_status': overall,
            'final_recommendation': final,
            'reason': reason,
        }

    def _find_drug_in_text(self, text: str) -> str:
        # naive lookup to find first known drug mentioned
            'suggestion': suggestion,
        if not text:
            return ''
        s = clean_text(text).lower()
        for p in self.dangerous_pairs.get('known_dangerous_pairs', []) + self.dangerous_pairs.get('known_safe_pairs', []):
            for key in ('drug1', 'drug2'):
                d = p.get(key, '').lower()
                if d and re.search(rf'\b{re.escape(d)}\b', s):
                    return d
        return ''


# global instance
_guardrails_instance: SafetyGuardrails = None


def get_guardrails(retriever=None) -> SafetyGuardrails:
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = SafetyGuardrails(retriever)
    return _guardrails_instance


if __name__ == '__main__':
    print_header('SAFETY GUARDRAILS - BASIC RUN')
    g = get_guardrails()
    resp = {'confidence': 0.92, 'citations': [{'text': 'Source: FDA Label'}], 'response': 'Metformin treats type 2 diabetes.'}
    chunks = [{'text': 'Metformin is indicated for type 2 diabetes.'}] * 8
    out = g.run_all_safety_checks(resp, 'What is metformin used for?', chunks)
    print(out)
