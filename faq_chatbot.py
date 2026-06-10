import json
import customtkinter as ctk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load FAQ Data
# -----------------------------

with open("faq_data.json", "r") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# -----------------------------
# NLP Processing
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)
question_vectors = vectorizer.fit_transform(questions)

# -----------------------------
# Chatbot Logic
# -----------------------------

def get_response(event=None):
    user_question = user_input.get().strip()

    if not user_question:
        return

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    print("Best Score:", best_score)
    print("Matched Question:", questions[best_match_index])

    if best_score < 0.15:
        response = (
            "Sorry, I couldn't understand your question. "
            "Please ask something related to our FAQs."
        )
    else:
        response = answers[best_match_index]

    chat_box.insert(
        "end",
        f"\n👤 You: {user_question}\n"
    )

    chat_box.insert(
        "end",
        f"🤖 Bot: {response}\n"
    )

    chat_box.see("end")

    user_input.delete(0, "end")


def clear_chat():
    chat_box.delete("1.0", "end")

    chat_box.insert(
        "end",
        "🤖 Bot: Hello! Ask me any question related to our FAQs.\n"
    )


# -----------------------------
# UI Settings
# -----------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("AI Powered FAQ Chatbot")
app.geometry("950x700")

# -----------------------------
# Title
# -----------------------------

title = ctk.CTkLabel(
    app,
    text="AI Powered FAQ Chatbot",
    font=("Arial", 30, "bold")
)

title.pack(pady=15)

# -----------------------------
# Chat Area
# -----------------------------

chat_box = ctk.CTkTextbox(
    app,
    width=850,
    height=500
)

chat_box.pack(pady=10)

chat_box.insert(
    "end",
    "🤖 Bot: Hello! Ask me any question related to our FAQs.\n"
)

# -----------------------------
# Input Frame
# -----------------------------

input_frame = ctk.CTkFrame(app)

input_frame.pack(
    pady=10,
    padx=10
)

user_input = ctk.CTkEntry(
    input_frame,
    width=600,
    placeholder_text="Type your question here..."
)

user_input.pack(
    side="left",
    padx=10,
    pady=10
)

# Press Enter to Send
user_input.bind(
    "<Return>",
    get_response
)

send_button = ctk.CTkButton(
    input_frame,
    text="Send",
    command=get_response,
    width=120
)

send_button.pack(
    side="left",
    padx=10
)

clear_button = ctk.CTkButton(
    input_frame,
    text="Clear Chat",
    command=clear_chat,
    width=120
)

clear_button.pack(
    side="left",
    padx=10
)

# -----------------------------
# Run App
# -----------------------------

app.mainloop()