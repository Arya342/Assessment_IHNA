from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/dashboard")
def redirect_to_streamlit():
    streamlit_url = "https://invoice1-px3qw3wmekhwsvdrzlzbzq.streamlit.app/"
    return RedirectResponse(url=streamlit_url)
