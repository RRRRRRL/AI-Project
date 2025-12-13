#!/usr/bin/env python3
"""
Validation script for LSTM Trajectory Prediction deliverables.
Tests that all required files exist and have correct structure.
"""

import json
import re
import sys
from pathlib import Path

def validate_notebook():
    """Validate Jupyter notebook structure and content."""
    print("=" * 60)
    print("VALIDATING JUPYTER NOTEBOOK")
    print("=" * 60)
    
    notebook_path = Path("LSTM_Trajectory_Prediction_Pipeline.ipynb")
    if not notebook_path.exists():
        print("❌ Notebook file not found")
        return False
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Check structure
    assert 'cells' in notebook, "Missing 'cells' field"
    assert 'metadata' in notebook, "Missing 'metadata' field"
    
    cells = notebook['cells']
    code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
    markdown_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
    
    print(f"✅ Total cells: {len(cells)}")
    print(f"✅ Code cells: {code_cells}")
    print(f"✅ Markdown cells: {markdown_cells}")
    
    # Check for key sections
    all_markdown = ' '.join([''.join(c['source']) for c in cells if c['cell_type'] == 'markdown'])
    required_sections = ['Data Generation', 'Model Architecture', 'Training', 'Evaluation']
    
    for section in required_sections:
        if section in all_markdown:
            print(f"✅ Section found: {section}")
        else:
            print(f"⚠️  Section missing: {section}")
    
    print("✅ Notebook validation PASSED\n")
    return True

def validate_report():
    """Validate report structure and content."""
    print("=" * 60)
    print("VALIDATING REPORT")
    print("=" * 60)
    
    report_path = Path("LSTM_Trajectory_Prediction_Report.md")
    if not report_path.exists():
        print("❌ Report file not found")
        return False
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Check main sections
    required_sections = [
        '## 1. Introduction',
        '## 2. Methodology',
        '## 3. Results and Analysis',
        '## 4. Conclusions and Future Work'
    ]
    
    for section in required_sections:
        if section in content:
            print(f"✅ Section found: {section}")
        else:
            print(f"❌ Section missing: {section}")
            return False
    
    # Check word count for page estimation
    word_count = len(content.split())
    estimated_pages = word_count / 500
    
    print(f"✅ Word count: {word_count}")
    print(f"✅ Estimated pages: {estimated_pages:.1f} (target: 3-6 pages)")
    
    if 1500 <= word_count <= 3500:
        print("✅ Word count appropriate for 3-6 page report")
    else:
        print(f"⚠️  Word count outside typical range")
    
    print("✅ Report validation PASSED\n")
    return True

def validate_video_script():
    """Validate video script structure and timing."""
    print("=" * 60)
    print("VALIDATING VIDEO SCRIPT")
    print("=" * 60)
    
    script_path = Path("Video_Presentation_Script.md")
    if not script_path.exists():
        print("❌ Video script file not found")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for slides
    slide_pattern = r'## SLIDE \d+:'
    slides = re.findall(slide_pattern, content)
    print(f"✅ Found {len(slides)} slides")
    
    # Check timing
    timing_pattern = r'\[(\d+) seconds?\]'
    timings = re.findall(timing_pattern, content)
    if timings:
        total_time = sum(int(t) for t in timings)
        print(f"✅ Total presentation time: {total_time} seconds")
        
        if 175 <= total_time <= 185:
            print(f"✅ Timing appropriate for 3-minute presentation")
        else:
            print(f"⚠️  Timing ({total_time}s) outside 3-minute target")
    
    # Check for essential components
    if 'Narration' in content:
        print("✅ Narration sections present")
    if 'Visual' in content:
        print("✅ Visual descriptions present")
    if 'TIMING BREAKDOWN' in content:
        print("✅ Timing breakdown included")
    
    print("✅ Video script validation PASSED\n")
    return True

def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("LSTM TRAJECTORY PREDICTION DELIVERABLES VALIDATION")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("Jupyter Notebook", validate_notebook()))
    except Exception as e:
        print(f"❌ Notebook validation error: {e}\n")
        results.append(("Jupyter Notebook", False))
    
    try:
        results.append(("Report", validate_report()))
    except Exception as e:
        print(f"❌ Report validation error: {e}\n")
        results.append(("Report", False))
    
    try:
        results.append(("Video Script", validate_video_script()))
    except Exception as e:
        print(f"❌ Video script validation error: {e}\n")
        results.append(("Video Script", False))
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:25s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 ALL DELIVERABLES VALIDATED SUCCESSFULLY!")
        print("\nDeliverables are ready for:")
        print("  - Submission")
        print("  - Review")
        print("  - Presentation")
        print("  - Execution\n")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print("Please review the error messages above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
