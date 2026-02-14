#!/usr/bin/env python3
"""
Extract SQL patterns and operators from queries.
Validates that Vietnamese translations properly reflect SQL operators.

This implements the Rule-Based Validation mentioned in the methodology:
- Aggregations: COUNT, SUM, AVG, MIN, MAX
- Comparisons: >, <, >=, <=, =, !=
- Logic: AND, OR, NOT
- Grouping: GROUP BY, HAVING, ORDER BY
- Set Operations: UNION, INTERSECT, EXCEPT
- Subqueries: Nested query structure

KEY VALIDATION: Checks if vi_question contains appropriate Vietnamese keywords
for each SQL operator (e.g., COUNT → "bao nhiêu", MAX → "lớn nhất")
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter


# Get project root (2 levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


# Vietnamese keyword mappings for SQL operators
VIETNAMESE_KEYWORDS = {
    'COUNT': [
        'bao nhiêu', 'mấy', 'số lượng', 'tổng số', 'có bao nhiêu',
        'có mấy', 'tính số', 'đếm'
    ],
    'SUM': [
        'tổng', 'cộng', 'tổng cộng', 'tổng số', 'tính tổng'
    ],
    'AVG': [
        'trung bình', 'tb', 'bình quân'
    ],
    'MAX': [
        'lớn nhất', 'cao nhất', 'nhiều nhất', 'tối đa', 'max',
        'lớn nhất', 'cao nhất', 'cực đại'
    ],
    'MIN': [
        'nhỏ nhất', 'thấp nhất', 'ít nhất', 'tối thiểu', 'min',
        'nhỏ nhất', 'thấp nhất', 'cực tiểu'
    ],
    'GREATER_THAN': [
        'lớn hơn', 'cao hơn', 'nhiều hơn', 'trên', 'vượt quá',
        'hơn', 'lớn hơn', 'cao hơn'
    ],
    'LESS_THAN': [
        'nhỏ hơn', 'thấp hơn', 'ít hơn', 'dưới', 'kém hơn',
        'dưới', 'nhỏ hơn', 'thấp hơn'
    ],
    'GREATER_EQUAL': [
        'lớn hơn hoặc bằng', 'từ', 'ít nhất', 'tối thiểu',
        'từ ... trở lên', 'không dưới'
    ],
    'LESS_EQUAL': [
        'nhỏ hơn hoặc bằng', 'tối đa', 'không quá', 'không vượt quá',
        'từ ... trở xuống', 'không trên'
    ],
    'EQUALS': [
        'bằng', 'là', 'có', 'được'
    ],
    'NOT_EQUALS': [
        'không', 'khác', 'không phải', 'khác với'
    ],
    'AND': [
        'và', 'cũng', 'đồng thời', 'vừa ... vừa'
    ],
    'OR': [
        'hoặc', 'hoặc là', 'hay'
    ],
    'NOT': [
        'không', 'không phải', 'chưa', 'ngoại trừ'
    ],
    'GROUP_BY': [
        'mỗi', 'từng', 'theo', 'cho mỗi', 'của mỗi',
        'theo từng', 'chia theo'
    ],
    'ORDER_BY': [
        'sắp xếp', 'xếp theo', 'theo thứ tự', 'thứ tự'
    ],
    'DISTINCT': [
        'khác nhau', 'riêng biệt', 'không trùng', 'duy nhất'
    ],
    'LIKE': [
        'chứa', 'có chứa', 'bắt đầu bằng', 'kết thúc bằng', 'giống'
    ],
    'IN': [
        'trong', 'thuộc', 'nằm trong'
    ],
    'BETWEEN': [
        'giữa', 'từ ... đến', 'trong khoảng'
    ],
}


def extract_sql_patterns(query: str) -> List[str]:
    """
    Extract SQL patterns and operators from a query.
    
    Args:
        query: SQL query string
        
    Returns:
        List of patterns found in the query
    """
    patterns = []
    query_upper = query.upper()
    
    # Basic structure
    if re.search(r'\bSELECT\b', query_upper):
        patterns.append('SELECT')
    
    if re.search(r'\bFROM\b', query_upper):
        patterns.append('FROM')
    
    if re.search(r'\bWHERE\b', query_upper):
        patterns.append('WHERE')
    
    # Joins
    if re.search(r'\bJOIN\b', query_upper):
        patterns.append('JOIN')
        
    if re.search(r'\bINNER\s+JOIN\b', query_upper):
        patterns.append('INNER_JOIN')
        
    if re.search(r'\bLEFT\s+JOIN\b', query_upper):
        patterns.append('LEFT_JOIN')
        
    if re.search(r'\bRIGHT\s+JOIN\b', query_upper):
        patterns.append('RIGHT_JOIN')
    
    # Aggregations
    aggregations = {
        'COUNT': r'\bCOUNT\s*\(',
        'SUM': r'\bSUM\s*\(',
        'AVG': r'\bAVG\s*\(',
        'MIN': r'\bMIN\s*\(',
        'MAX': r'\bMAX\s*\(',
    }
    
    for agg_name, agg_pattern in aggregations.items():
        if re.search(agg_pattern, query_upper):
            patterns.append(agg_name)
    
    # Grouping and ordering
    if re.search(r'\bGROUP\s+BY\b', query_upper):
        patterns.append('GROUP_BY')
    
    if re.search(r'\bHAVING\b', query_upper):
        patterns.append('HAVING')
    
    if re.search(r'\bORDER\s+BY\b', query_upper):
        patterns.append('ORDER_BY')
    
    # Sorting direction
    if re.search(r'\bDESC\b', query_upper):
        patterns.append('DESC')
        
    if re.search(r'\bASC\b', query_upper):
        patterns.append('ASC')
    
    # Comparison operators
    comparisons = {
        'EQUALS': r'=',
        'NOT_EQUALS': r'!=|<>',
        'GREATER_THAN': r'>',
        'LESS_THAN': r'<',
        'GREATER_EQUAL': r'>=',
        'LESS_EQUAL': r'<=',
        'LIKE': r'\bLIKE\b',
        'IN': r'\bIN\s*\(',
        'BETWEEN': r'\bBETWEEN\b',
    }
    
    for comp_name, comp_pattern in comparisons.items():
        if re.search(comp_pattern, query_upper):
            patterns.append(comp_name)
    
    # Logic operators
    if re.search(r'\bAND\b', query_upper):
        patterns.append('AND')
        
    if re.search(r'\bOR\b', query_upper):
        patterns.append('OR')
        
    if re.search(r'\bNOT\b', query_upper):
        patterns.append('NOT')
    
    # Set operations
    if re.search(r'\bUNION\b', query_upper):
        patterns.append('UNION')
        
    if re.search(r'\bINTERSECT\b', query_upper):
        patterns.append('INTERSECT')
        
    if re.search(r'\bEXCEPT\b', query_upper):
        patterns.append('EXCEPT')
    
    # Subqueries (count nested SELECT statements)
    select_count = len(re.findall(r'\bSELECT\b', query_upper))
    if select_count > 1:
        patterns.append('SUBQUERY')
        if select_count > 2:
            patterns.append('NESTED_SUBQUERY')
    
    # Distinct
    if re.search(r'\bDISTINCT\b', query_upper):
        patterns.append('DISTINCT')
    
    # Limit
    if re.search(r'\bLIMIT\b', query_upper):
        patterns.append('LIMIT')
    
    return patterns


def categorize_complexity(patterns: List[str]) -> str:
    """
    Categorize query complexity based on patterns.
    
    Args:
        patterns: List of SQL patterns
        
    Returns:
        Complexity category: basic, intermediate, advanced, expert
    """
    # Expert: nested subqueries or set operations
    if 'NESTED_SUBQUERY' in patterns or any(op in patterns for op in ['UNION', 'INTERSECT', 'EXCEPT']):
        return 'expert'
    
    # Advanced: subqueries or complex joins with aggregations
    if 'SUBQUERY' in patterns:
        return 'advanced'
    
    agg_count = sum(1 for p in patterns if p in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX'])
    if agg_count > 1 and 'GROUP_BY' in patterns:
        return 'advanced'
    
    # Intermediate: joins or aggregations with grouping
    if 'JOIN' in patterns or ('GROUP_BY' in patterns and agg_count > 0):
        return 'intermediate'
    
    # Basic: simple SELECT
    return 'basic'


def validate_vietnamese_translation(vi_question: str, sql_patterns: List[str]) -> Dict:
    """
    Validate if Vietnamese translation contains appropriate keywords for SQL operators.
    
    Args:
        vi_question: Vietnamese translated question
        sql_patterns: List of SQL patterns extracted from query
        
    Returns:
        Dictionary with validation results
    """
    vi_lower = vi_question.lower()
    
    validation_results = {
        'is_valid': True,
        'missing_operators': [],
        'matched_operators': [],
        'warnings': []
    }
    
    # Check critical operators that MUST have Vietnamese equivalents
    critical_operators = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 
                          'GREATER_THAN', 'LESS_THAN', 'GREATER_EQUAL', 'LESS_EQUAL']
    
    for pattern in sql_patterns:
        if pattern not in VIETNAMESE_KEYWORDS:
            continue
        
        keywords = VIETNAMESE_KEYWORDS[pattern]
        found = any(keyword in vi_lower for keyword in keywords)
        
        if found:
            validation_results['matched_operators'].append(pattern)
        elif pattern in critical_operators:
            # Critical operator missing
            validation_results['missing_operators'].append(pattern)
            validation_results['is_valid'] = False
        else:
            # Non-critical operator (may be implicit in Vietnamese)
            validation_results['warnings'].append(pattern)
    
    return validation_results


def get_validation_score(validation: Dict) -> float:
    """
    Calculate validation score based on operator matching.
    
    Args:
        validation: Validation results dictionary
        
    Returns:
        Score from 0.0 to 1.0
    """
    total_checked = len(validation['matched_operators']) + len(validation['missing_operators'])
    
    if total_checked == 0:
        return 1.0  # No operators to check
    
    matched = len(validation['matched_operators'])
    return matched / total_checked


def analyze_dataset(data: List[Dict]) -> Dict:
    """
    Analyze SQL patterns across the entire dataset.
    
    Args:
        data: List of samples
        
    Returns:
        Analysis results dictionary
    """
    all_patterns = []
    complexity_dist = Counter()
    validation_scores = []
    failed_validations = []
    
    for sample in data:
        patterns = sample.get('sql_patterns', [])
        all_patterns.extend(patterns)
        
        complexity = sample.get('sql_complexity', 'unknown')
        complexity_dist[complexity] += 1
        
        # Validation analysis
        validation = sample.get('operator_validation', {})
        score = sample.get('operator_validation_score', 0.0)
        validation_scores.append(score)
        
        if not validation.get('is_valid', True):
            failed_validations.append({
                'id': sample['id'],
                'score': score,
                'missing': validation.get('missing_operators', []),
                'vi_question': sample.get('vi_question', '')[:80]
            })
    
    pattern_counts = Counter(all_patterns)
    
    # Validation statistics
    avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
    perfect_score = sum(1 for s in validation_scores if s >= 1.0)
    good_score = sum(1 for s in validation_scores if 0.8 <= s < 1.0)
    poor_score = sum(1 for s in validation_scores if s < 0.8)
    
    return {
        'total_samples': len(data),
        'pattern_distribution': dict(pattern_counts.most_common()),
        'complexity_distribution': dict(complexity_dist),
        'unique_patterns': len(pattern_counts),
        'most_common_patterns': pattern_counts.most_common(10),
        'validation_stats': {
            'avg_score': avg_validation_score,
            'perfect_validations': perfect_score,
            'good_validations': good_score,
            'poor_validations': poor_score,
            'failed_samples': len(failed_validations),
            'failed_details': failed_validations[:20]  # Top 20 failures
        }
    }


def main():
    """Main execution function."""
    print("="*80)
    print("SQL PATTERN EXTRACTION & VIETNAMESE VALIDATION")
    print("="*80)
    print()
    
    # Load data
    input_file = PROJECT_ROOT / 'data/manual_translations/vispider_train_2000.json'
    
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} samples\n")
    
    # Extract patterns and validate for each sample
    print("Extracting SQL patterns and validating Vietnamese translations...")
    
    for sample in data:
        query = sample.get('query', '')
        vi_question = sample.get('vi_question', '')
        
        # Extract patterns
        patterns = extract_sql_patterns(query)
        sample['sql_patterns'] = patterns
        
        # Categorize complexity
        complexity = categorize_complexity(patterns)
        sample['sql_complexity'] = complexity
        
        # Validate Vietnamese translation
        validation = validate_vietnamese_translation(vi_question, patterns)
        sample['operator_validation'] = validation
        
        # Calculate validation score
        score = get_validation_score(validation)
        sample['operator_validation_score'] = round(score, 4)
    
    print("✓ Pattern extraction and validation complete\n")
    
    # Analyze results
    print("="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    print()
    
    analysis = analyze_dataset(data)
    
    print(f"Total samples analyzed: {analysis['total_samples']}")
    print(f"Unique patterns found: {analysis['unique_patterns']}")
    print()
    
    # Pattern distribution
    print("Top 20 SQL patterns:")
    print(f"{'Pattern':<20} {'Count':<8} {'Percentage':<10}")
    print("-"*80)
    
    for pattern, count in analysis['most_common_patterns'][:20]:
        pct = count / analysis['total_samples'] * 100
        print(f"{pattern:<20} {count:<8} {pct:>6.1f}%")
    
    print()
    
    # Complexity distribution
    print("SQL Complexity Distribution:")
    print(f"{'Complexity':<15} {'Count':<8} {'Percentage':<10}")
    print("-"*80)
    
    for complexity in ['basic', 'intermediate', 'advanced', 'expert']:
        count = analysis['complexity_distribution'].get(complexity, 0)
        pct = count / analysis['total_samples'] * 100
        print(f"{complexity:<15} {count:<8} {pct:>6.1f}%")
    
    print()
    
    # Pattern co-occurrence analysis
    print("="*80)
    print("PATTERN INSIGHTS")
    print("="*80)
    print()
    
    # Aggregation usage
    agg_patterns = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    agg_counts = {p: analysis['pattern_distribution'].get(p, 0) for p in agg_patterns}
    total_agg = sum(agg_counts.values())
    
    print(f"Aggregation Functions ({total_agg} total occurrences):")
    for pattern, count in agg_counts.items():
        if count > 0:
            pct = count / total_agg * 100
            print(f"  {pattern:<10}: {count:4d} ({pct:5.1f}%)")
    
    print()
    
    # Join usage
    join_count = analysis['pattern_distribution'].get('JOIN', 0)
    print(f"Queries with JOINs: {join_count} ({join_count/analysis['total_samples']*100:.1f}%)")
    
    # Subquery usage
    subquery_count = analysis['pattern_distribution'].get('SUBQUERY', 0)
    nested_count = analysis['pattern_distribution'].get('NESTED_SUBQUERY', 0)
    print(f"Queries with subqueries: {subquery_count} ({subquery_count/analysis['total_samples']*100:.1f}%)")
    print(f"Queries with nested subqueries: {nested_count} ({nested_count/analysis['total_samples']*100:.1f}%)")
    
    print()
    
    # VALIDATION ANALYSIS
    print("="*80)
    print("VIETNAMESE TRANSLATION VALIDATION")
    print("="*80)
    print()
    
    val_stats = analysis['validation_stats']
    
    print(f"Average validation score: {val_stats['avg_score']:.4f}")
    print(f"Perfect validations (1.0):   {val_stats['perfect_validations']:4d} ({val_stats['perfect_validations']/analysis['total_samples']*100:5.1f}%)")
    print(f"Good validations (0.8-1.0):  {val_stats['good_validations']:4d} ({val_stats['good_validations']/analysis['total_samples']*100:5.1f}%)")
    print(f"Poor validations (< 0.8):    {val_stats['poor_validations']:4d} ({val_stats['poor_validations']/analysis['total_samples']*100:5.1f}%)")
    print()
    
    if val_stats['failed_samples'] > 0:
        print(f"⚠️  {val_stats['failed_samples']} samples have missing SQL operator keywords in Vietnamese")
        print()
        print("Top 20 samples with operator validation failures:")
        print(f"{'ID':<15} {'Score':<8} {'Missing Operators':<40} {'Vietnamese Question'}")
        print("-"*120)
        
        for fail in val_stats['failed_details']:
            operators = ', '.join(fail['missing'])
            print(f"{fail['id']:<15} {fail['score']:<8.4f} {operators:<40} {fail['vi_question']}")
        
        print()
        print("These samples should be reviewed to ensure SQL operators are properly translated.")
    else:
        print("✅ All samples have proper Vietnamese keywords for SQL operators!")
    
    print()
    
    # Save analysis report
    analysis_file = PROJECT_ROOT / 'results/quality_analysis/sql_pattern_analysis.json'
    
    print(f"Saving analysis report to: {analysis_file}")
    
    # Prepare analysis with sample data
    analysis_with_samples = analysis.copy()
    analysis_with_samples['annotated_samples'] = data
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_with_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved analysis report with annotated samples")
    
    print()
    print("="*80)
    print("✅ SQL PATTERN EXTRACTION COMPLETE")
    print("="*80)
    print()
    
    print("Output files:")
    print(f"  1. {analysis_file}")
    print(f"     → Pattern analysis, validation stats, and annotated samples")
    
    # Export failed validation IDs for manual review
    if val_stats['failed_samples'] > 0:
        failed_ids_file = PROJECT_ROOT / 'results/quality_analysis/operator_validation_failures_ids.txt'
        
        failed_samples = [
            s for s in data 
            if not s.get('operator_validation', {}).get('is_valid', True)
        ]
        
        # Export IDs to text file
        with open(failed_ids_file, 'w', encoding='utf-8') as f:
            for sample in failed_samples:
                f.write(f"{sample['id']}\n")
        
        print(f"  2. {failed_ids_file}")
        print(f"     → {len(failed_samples)} sample IDs needing operator validation review")
        
        # Also save full details for reference
        failed_file = PROJECT_ROOT / 'results/quality_analysis/operator_validation_failures.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        
        print(f"  3. {failed_file}")
        print(f"     → Full details of failed validations (for reference)")
    
    print()
    
    print("Next steps:")
    print("  - Review samples with operator validation failures")
    print("  - Ensure COUNT/MAX/MIN/comparison operators are properly translated")
    print("  - Use sql_patterns for query type stratification")
    print("  - Validate pattern coverage across difficulty levels")
    print("  - Use sql_complexity for sampling strategy")


if __name__ == "__main__":
    main()
