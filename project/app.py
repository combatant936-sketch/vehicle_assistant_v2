import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import project.versions.v2.rag as rag
import project.db_setup.db as db
from typing import Optional, Any,Literal

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    conversation_id: str
    question: str
    answera: str

class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: Literal[1, -1]

@app.post("/question", response_model=AnswerResponse)
def handle_question(request: QuestionRequest):
    question = request.question

    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    conversation_id = str(uuid.uuid4())
    result=rag.query(question)

    # print("########################",result)
    

    db.save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer_data=result,
    )

    return AnswerResponse(
        conversation_id=conversation_id,
        question=question,
        answera=result["answer"],
    )



@app.post("/feedback")
def handle_feedback(request: FeedbackRequest):
    db.save_feedback(
        conversation_id=request.conversation_id,
        feedback=request.feedback,
    )
    return {"message": f"Feedback received: {request.feedback}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)