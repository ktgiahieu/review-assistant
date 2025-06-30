
# ICLR 2024 + o4-mini
# python generate_checklistanswer.py --input_dir data/ICLR2024_latest/accepted --output_dir data/ICLR2024_run1/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 200 --deployment_name Checklist-GPT-o4-mini --debug_timer
# python generate_checklistanswer.py --input_dir data/ICLR2024_latest/rejected --output_dir data/ICLR2024_run1/rejected --file_pattern "./*/structured_paper_output/paper.md" --max_workers 200 --deployment_name Checklist-GPT-o4-mini --debug_timer

# ICLR 2024 + o3
python generate_checklistanswer.py --input_dir data/ICLR2024_latest/accepted --output_dir data/ICLR2024_run1/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 200 --deployment_name Checklist-GPT-o3 --debug_timer
python generate_checklistanswer.py --input_dir data/ICLR2024_latest/rejected --output_dir data/ICLR2024_run1/rejected --file_pattern "./*/structured_paper_output/paper.md" --max_workers 200 --deployment_name Checklist-GPT-o3 --debug_timer