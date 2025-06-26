# Generate Checklist Answer

To generate:

- Put your data (e.g. `ICLR2024_latest`) in `./data`:
- Create a `.env` file with these info to run Azure OpenAI API:

  - KEY = `<YOUR KEY>`
  - API_VERSION = 2024-12-01-preview
  - LOCATION = eastus
  - ENDPOINT = `<YOUR ENDPOINT>`
- Modify the script `./generate_checklist_script.sh` with the correct config and run:

```bash
./generate_checklist_script.sh 
```

Prerequisites:

- openai
- Pillow
