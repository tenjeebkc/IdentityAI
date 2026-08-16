** Steps to start the app **

** Terminal 1: Start Ollama **
ollama run llama3.2   (or any other model)

** Terminal 2: Start the FastAPI backend **
uvicorn server:app --reload  (inside the backend folder)

** If you changed or edited your knowledge base **
python3 create_db.py

** After it's done, restart the backend **

** Finally, open the frontend and start a conversation **
