#!/usr/bin/env python3
"""
Translate samples using GPT with diverse few-shot prompting and real-time validation.

Features:
1. Intra-class diversity: Select N diverse examples with same SQL pattern
2. Real-time validation:
   - LaBSE similarity >= 0.75 (0.70 for short questions <= 7 words)
   - Operator validation: COUNT only checked when question explicitly asks "how many"
3. Automatic retry (2 attempts total):
   - Attempt 1: 3 examples, standard prompt
   - Attempt 2: 5 completely new examples (exclude attempt 1) + literal-translation hint
4. Checkpoint: Save progress every 100 samples
5. Rate limiting: Respect OpenAI API limits
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dotenv import load_dotenv

# Get project root first
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Import validation functions
import sys
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/utils'))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    print("Run: pip install openai")
    sys.exit(1)


# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')


def load_gold_seed_with_embeddings() -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Load gold seed data with LaBSE embeddings."""
    print("Loading gold seed data and embeddings...")
    
    # Load pattern analysis (contains annotated samples)
    pattern_file = PROJECT_ROOT / 'results/quality_analysis/sql_pattern_analysis.json'
    with open(pattern_file, 'r', encoding='utf-8') as f:
        pattern_data = json.load(f)
    
    gold_samples = pattern_data['annotated_samples']
    
    # Load embeddings from vispider_embeddings.json (same order as gold_samples)
    embeddings_file = PROJECT_ROOT / 'data/manual_translations/vispider_embeddings.json'
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embeddings_raw = json.load(f)['embeddings']
    
    # Match by position (both lists have 1996 samples in same order)
    embeddings_dict = {}
    for i, (sample, emb_item) in enumerate(zip(gold_samples, embeddings_raw)):
        emb = np.array(emb_item['en_embedding'])
        norm = np.linalg.norm(emb)
        embeddings_dict[sample['id']] = emb / norm if norm > 0 else emb
    
    print(f"✓ Loaded {len(gold_samples)} gold samples with embeddings\n")
    
    return gold_samples, embeddings_dict


def build_pattern_index(gold_samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Build index of samples by their primary SQL pattern."""
    pattern_index = defaultdict(list)
    
    for sample in gold_samples:
        patterns = sample.get('sql_patterns', ['SELECT'])
        primary_pattern = patterns[0] if patterns else 'SELECT'
        pattern_index[primary_pattern].append(sample)
    
    print(f"Built pattern index with {len(pattern_index)} patterns")
    
    for pattern in sorted(pattern_index.keys())[:10]:
        print(f"  {pattern}: {len(pattern_index[pattern])} samples")
    
    print()
    
    return dict(pattern_index)


def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine distance between two embeddings."""
    return 1.0 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# Retry strategies: (n_examples, extra_instruction)
# Attempt 1: 3 examples, standard prompt
# Attempt 2: 5 completely different examples + hint to be more literal
RETRY_STRATEGIES = [
    (
        3,
        None
    ),
    (
        5,
        "IMPORTANT: Your previous translation scored low on semantic similarity. "
        "Translate more literally and directly. "
        "If the SQL contains COUNT/MAX/MIN/AVG/SUM or comparisons (>, <), "
        "make sure the Vietnamese includes: "
        "COUNT \u2192 'bao nhi\u00eau' or 's\u1ed1 l\u01b0\u1ee3ng'; "
        "MAX \u2192 'l\u1edbn nh\u1ea5t'; MIN \u2192 'nh\u1ecf nh\u1ea5t'; "
        "AVG \u2192 'trung b\u00ecnh'; SUM \u2192 't\u1ed5ng'; "
        "greater-than \u2192 'l\u1edbn h\u01a1n'; less-than \u2192 'nh\u1ecf h\u01a1n'."
    ),
]


def select_diverse_fewshot(
    target_sample: Dict,
    pattern_index: Dict[str, List[Dict]],
    embeddings_dict: Dict[str, np.ndarray],
    exclude_ids: Set[str] = None,
    n_examples: int = 3
) -> List[Dict]:
    """
    Select n_examples diverse few-shot examples with same primary pattern.
    
    Greedy selection for max diversity:
    - First: random from pattern pool
    - Each subsequent: max min-distance from already-selected set
    """
    if exclude_ids is None:
        exclude_ids = set()
    
    # Get target pattern
    patterns = target_sample.get('sql_patterns', ['SELECT'])
    primary_pattern = patterns[0] if patterns else 'SELECT'
    
    # Get candidate pool
    if primary_pattern not in pattern_index:
        primary_pattern = list(pattern_index.keys())[0]
    
    candidates = [
        s for s in pattern_index[primary_pattern] 
        if s['id'] not in exclude_ids and s['id'] in embeddings_dict
    ]
    
    if len(candidates) < n_examples:
        # Not enough same-pattern candidates, expand to all patterns
        candidates = [
            s for samples in pattern_index.values()
            for s in samples
            if s['id'] not in exclude_ids and s['id'] in embeddings_dict
        ]
    
    if len(candidates) < n_examples:
        raise ValueError(f"Not enough candidates for few-shot selection (only {len(candidates)} available)")
    
    # Greedy diverse selection
    selected = []
    
    # First: random
    first = np.random.choice(candidates)
    selected.append(first)
    remaining = [c for c in candidates if c['id'] != first['id']]
    
    # Each next: max min-distance from all already-selected
    while len(selected) < n_examples and remaining:
        selected_embs = [embeddings_dict[s['id']] for s in selected]
        min_dists = []
        for c in remaining:
            c_emb = embeddings_dict[c['id']]
            min_d = min(cosine_distance(se, c_emb) for se in selected_embs)
            min_dists.append(min_d)
        next_idx = np.argmax(min_dists)
        selected.append(remaining[next_idx])
        remaining = [c for c in remaining if c['id'] != remaining[next_idx]['id']]
    
    return selected


def create_translation_prompt(target: Dict, fewshot_examples: List[Dict], extra_instruction: str = None) -> str:
    """Create GPT prompt with few-shot examples using template file."""
    
    # Load prompt template
    template_path = Path(__file__).parent / "prompt_template.txt"
    template_content = template_path.read_text()
    
    # Format examples
    examples_text = ""
    for i, example in enumerate(fewshot_examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Database: {example.get('db_id', 'N/A')}\n"
        examples_text += f"English: {example['question']}\n"
        examples_text += f"SQL: {example['query']}\n"
        examples_text += f"Vietnamese: {example['vi_question']}\n\n"
    
    # Format prompt with all fields
    prompt = template_content.format(
        examples=examples_text.strip(),
        database=target.get('db_id', 'N/A'),
        english_question=target['question'],
        sql_query=target['query']
    )
    
    # Inject extra instruction before "NOW TRANSLATE:" if provided
    if extra_instruction:
        prompt = prompt.replace(
            "NOW TRANSLATE:",
            f"⚠️ {extra_instruction}\n\nNOW TRANSLATE:"
        )
    
    return prompt


# Reasoning models that use the Responses API with reasoning= parameter
_REASONING_MODELS = {'o1', 'o1-mini', 'o3', 'o3-mini', 'o4-mini', 'gpt-5-mini'}


def _is_reasoning_model(model: str) -> bool:
    """Return True if model uses the Responses API with reasoning parameter."""
    return any(model.startswith(m) or model == m for m in _REASONING_MODELS)


def call_gpt_translate(client: OpenAI, prompt: str, model: str) -> str:
    """Call GPT API for translation.

    - Reasoning models (o-series, gpt-5-mini): use Responses API with reasoning.
    - Standard models (gpt-4o, gpt-4o-mini, etc.): use Chat Completions API.
    """
    try:
        if _is_reasoning_model(model):
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}]
                    }
                ],
                text={"format": {"type": "text"}},
                reasoning={"effort": "low"},
                store=False,
                include=[],
            )
            translation = response.output_text.strip()
        else:
            # Standard Chat Completions API
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            translation = response.choices[0].message.content.strip()

        return translation

    except Exception as e:
        print(f"  ❌ GPT API error: {e}")
        return None


def validate_translation(
    english_question: str,
    vietnamese_translation: str,
    sql_patterns: List[str]
) -> Tuple[bool, float, Dict]:
    """
    Validate translation quality.
    
    Returns:
        (is_valid, labse_score, operator_validation)
    """
    from embeddings_utils import compute_similarity
    
    # Compute LaBSE similarity
    try:
        labse_score = compute_similarity(english_question, vietnamese_translation)
        
    except Exception as e:
        print(f"  ⚠️  LaBSE error: {e}")
        labse_score = 0.0
    
    # Operator validation (simplified keywords check)
    vi_lower = vietnamese_translation.lower()
    en_lower = english_question.lower()
    
    missing_operators = []
    
    # COUNT: only require explicit count keywords when the question is clearly
    # asking "how many". If the question asks "which/what X is most common",
    # COUNT is implicit in SQL but doesn't need to appear in Vietnamese.
    count_question_words = ['how many', 'how much', 'count', 'number of', 'total number']
    count_is_explicit = any(w in en_lower for w in count_question_words)
    
    critical_mappings = {
        'MAX': ['l\u1edbn nh\u1ea5t', 'cao nh\u1ea5t', 'nhi\u1ec1u nh\u1ea5t', 't\u1ed1i \u0111a'],
        'MIN': ['nh\u1ecf nh\u1ea5t', 'th\u1ea5p nh\u1ea5t', '\u00edt nh\u1ea5t', 't\u1ed1i thi\u1ec3u'],
        'AVG': ['trung b\u00ecnh', 'b\u00ecnh qu\u00e2n'],
        'SUM': ['t\u1ed5ng', 't\u1ed5ng c\u1ed9ng'],
        'GREATER_THAN': ['l\u1edbn h\u01a1n', 'cao h\u01a1n', 'nhi\u1ec1u h\u01a1n', 'tr\u00ean', 'v\u01b0\u1ee3t'],
        'LESS_THAN': ['nh\u1ecf h\u01a1n', 'th\u1ea5p h\u01a1n', '\u00edt h\u01a1n', 'd\u01b0\u1edbi'],
    }
    # Only add COUNT to mapping when question explicitly asks for a count
    if count_is_explicit:
        critical_mappings['COUNT'] = ['bao nhi\u00eau', 'm\u1ea5y', 's\u1ed1 l\u01b0\u1ee3ng', '\u0111\u1ebfm']
    
    for pattern in sql_patterns:
        if pattern in critical_mappings:
            keywords = critical_mappings[pattern]
            if not any(kw in vi_lower for kw in keywords):
                missing_operators.append(pattern)
    
    operator_valid = len(missing_operators) == 0
    
    # Dynamic LaBSE threshold: short questions are inherently harder for
    # cross-lingual embeddings, so lower the bar slightly
    en_word_count = len(english_question.split())
    labse_threshold = 0.70 if en_word_count <= 7 else 0.75
    
    # Overall validation
    is_valid = labse_score >= labse_threshold and operator_valid
    
    return is_valid, labse_score, {
        'is_valid': operator_valid,
        'missing_operators': missing_operators
    }


def save_checkpoint(
    results: List[Dict],
    checkpoint_num: int,
    output_dir: Path
):
    """Save checkpoint of translation results."""
    checkpoint_file = output_dir / f'gpt_translations_checkpoint_{checkpoint_num:04d}.json'
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 Checkpoint saved: {checkpoint_file.name}")


def main():
    """Main execution function."""
    print("="*80)
    print("GPT TRANSLATION WITH VALIDATION")
    print("="*80)
    print()

    # Check environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('GPT_MODEL', 'gpt-4o-mini')

    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("Please add your OpenAI API key to .env")
        return

    print(f"Using model: {model}")
    print(f"API key: {api_key[:20]}...")
    print()

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Load gold seed + embeddings
    gold_samples, embeddings_dict = load_gold_seed_with_embeddings()
    pattern_index = build_pattern_index(gold_samples)

    # Load target samples (output of script 01)
    target_file = PROJECT_ROOT / 'data/chatgpt_translations/gpt_target_samples.json'
    if not target_file.exists():
        print(f"❌ ERROR: Target file not found: {target_file}")
        print("Please run first: python3 scripts/phase3_chatgpt/01_select_samples_for_gpt.py")
        return

    print(f"Loading target samples from: {target_file}")
    with open(target_file, 'r', encoding='utf-8') as f:
        target_samples = json.load(f)
    print(f"✓ Loaded {len(target_samples)} target samples\n")

    # Output directory
    output_dir = PROJECT_ROOT / 'data/chatgpt_translations'
    output_dir.mkdir(exist_ok=True)

    # ── Resume from checkpoint ────────────────────────────────────────────
    results: List[Dict] = []
    failed_samples: List[Dict] = []
    done_ids: Set[str] = set()

    checkpoint_files = sorted(output_dir.glob('gpt_translations_checkpoint_*.json'))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"🔄 Resuming from checkpoint: {latest_ckpt.name}")
        with open(latest_ckpt, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_ids = {r['id'] for r in results}
        print(f"   Already translated: {len(done_ids)} samples — skipping these.\n")
    else:
        print("No checkpoint found — starting from scratch.\n")

    # Translation loop
    print("="*80)
    print("TRANSLATION PROGRESS")
    print("="*80)
    print()

    start_time = time.time()

    for idx, target in enumerate(target_samples, 1):
        # Skip already-translated samples when resuming
        if target['id'] in done_ids:
            continue
        print(f"[{idx}/{len(target_samples)}] {target['id']}")
        
        success = False
        attempts = 0
        max_attempts = len(RETRY_STRATEGIES)
        excluded_example_ids = set()
        
        while attempts < max_attempts and not success:
            strategy = RETRY_STRATEGIES[attempts]
            n_examples, extra_instruction = strategy
            attempts += 1
            
            try:
                # Select diverse few-shot examples
                fewshot_examples = select_diverse_fewshot(
                    target,
                    pattern_index,
                    embeddings_dict,
                    exclude_ids=excluded_example_ids,
                    n_examples=n_examples
                )
                
                # Create prompt
                prompt = create_translation_prompt(target, fewshot_examples, extra_instruction)
                
                # Call GPT
                translation = call_gpt_translate(client, prompt, model)
                
                if translation is None:
                    print(f"  ⚠️  Attempt {attempts}: API call failed, retrying...")
                    time.sleep(2)
                    continue
                
                # Validate
                is_valid, labse_score, operator_validation = validate_translation(
                    target['question'],
                    translation,
                    target.get('sql_patterns', [])
                )
                
                hint_label = f" ({n_examples} examples" + (", +hint)" if extra_instruction else ")")
                print(f"  Translation: {translation[:60]}...")
                print(f"  LaBSE: {labse_score:.4f} | Operators: {'✓' if operator_validation['is_valid'] else '✗'} | Attempt {attempts}{hint_label}")
                
                if is_valid:
                    result = target.copy()
                    result['vi_question'] = translation
                    result['labse_similarity'] = labse_score
                    result['operator_validation'] = operator_validation
                    result['fewshot_examples_used'] = [e['id'] for e in fewshot_examples]
                    result['attempts'] = attempts
                    
                    results.append(result)
                    success = True
                    print(f"  ✅ Success (attempt {attempts})")
                else:
                    print(f"  ⚠️  Attempt {attempts}: Validation failed (LaBSE={labse_score:.4f}, operators={'ok' if operator_validation['is_valid'] else 'missing'})")
                    # Exclude used examples for next attempt
                    excluded_example_ids.update([e['id'] for e in fewshot_examples])
                    time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Error on attempt {attempts}: {e}")
                time.sleep(2)
        
        if not success:
            print(f"  ❌ Failed after {max_attempts} attempts (strategies exhausted)")
            failed_entry = target.copy()
            failed_entry['failure_reason'] = (
                f"Both attempts failed: "
                f"attempt 1 ({RETRY_STRATEGIES[0][0]} examples), "
                f"attempt 2 ({RETRY_STRATEGIES[1][0]} new examples + hint)"
            )
            failed_samples.append(failed_entry)
        
        print()
        
        # Checkpoint every 100 samples
        if idx % 100 == 0:
            save_checkpoint(results, idx // 100, output_dir)
            
            # Print progress stats
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (len(target_samples) - idx) / rate
            
            print(f"  📊 Progress: {idx}/{len(target_samples)} ({idx/len(target_samples)*100:.1f}%)")
            print(f"  ⏱️  Rate: {rate:.2f} samples/sec | ETA: {remaining/60:.1f} minutes")
            print()
        
        # Rate limiting (avoid hitting API limits)
        time.sleep(0.5)  # 2 requests per second max
    
    # Save final results
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    print()

    final_file = output_dir / 'gpt_translations_final.json'
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(results)} successful translations to: {final_file}")

    # Save failed samples log
    failed_file = PROJECT_ROOT / 'results/quality_analysis/gpt_failed_samples.json'
    if failed_samples:
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"⚠️  Saved {len(failed_samples)} failed samples to: {failed_file}")
    else:
        if failed_file.exists():
            failed_file.unlink()
        print("✅ No failed samples — all translated successfully")
    
    # Generate validation report
    labse_scores = [r['labse_similarity'] for r in results]
    operator_valid_count = sum(1 for r in results if r['operator_validation']['is_valid'])
    
    report = {
        'total_target': len(target_samples),
        'successful': len(results),
        'failed': len(failed_samples),
        'success_rate': len(results) / len(target_samples) * 100,
        'labse_stats': {
            'mean': float(np.mean(labse_scores)),
            'median': float(np.median(labse_scores)),
            'min': float(np.min(labse_scores)),
            'max': float(np.max(labse_scores)),
        },
        'operator_validation_rate': operator_valid_count / len(results) * 100,
    }
    
    report_file = PROJECT_ROOT / 'results/quality_analysis/gpt_validation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved validation report to: {report_file}")
    print()
    
    # Print summary
    print("="*80)
    print("✅ TRANSLATION COMPLETE")
    print("="*80)
    print()
    
    print(f"Success rate: {report['success_rate']:.1f}% ({len(results)}/{len(target_samples)})")
    print(f"LaBSE mean: {report['labse_stats']['mean']:.4f}")
    print(f"Operator validation: {report['operator_validation_rate']:.1f}%")
    print()
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/len(target_samples):.2f} seconds/sample")


if __name__ == "__main__":
    main()
