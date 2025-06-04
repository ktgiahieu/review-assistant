# Generate Checklist Answer

To generate:

```bash
python3 generate_checklistanswer.py --input_dir data/ICLR2024/accepted --output_dir data/ICLR2024/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 40 --deployment_name Checklist-GPT-o4-mini --debug_timer   
```

Prerequisites:

- openai
- Pillow
