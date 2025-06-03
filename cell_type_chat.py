import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Using base GPT-2 for now
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Few-shot prompt template
FEW_SHOT_PROMPT = """Q: Cluster 0 is enriched with IL7R. What cell type is this?
A: CD4 T cells

Q: Cluster 1 is enriched with CD14, LYZ. What cell type is this?
A: CD14+ Monocytes

Q: Cluster 2 is enriched with MS4A1. What cell type is this?
A: B cells

Q: Cluster 3 is enriched with CD8A. What cell type is this?
A: CD8 T cells

Q: Cluster 4 is enriched with GNLY, NKG7. What cell type is this?
A: NK cells

Q: Cluster 5 is enriched with FCGR3A, MS4A7. What cell type is this?
A: FCGR3A+ Monocytes

Q: Cluster 6 is enriched with FCER1A, CST3. What cell type is this?
A: Dendritic Cells

Q: Cluster 7 is enriched with PPBP. What cell type is this?
A: Megakaryocytes

"""

def generate_response(prompt, model, tokenizer, max_new_tokens=50):
    # Combine few-shot examples with user input
    full_prompt = FEW_SHOT_PROMPT + prompt + "\nA:"
    
    # Tokenize input
    inputs = tokenizer.encode(full_prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("A:")[-1].strip()
    return response

def main():
    st.title("Cell Type Annotation Chat")
    st.write("Ask about cell types based on gene markers")
    
    # Add sample questions as simple text
    st.markdown("""
    **Sample Questions:**
    
    - Cluster X is enriched with IL7R. What cell type is this?
    - What cell type is associated with CD14 and LYZ markers?
    - I found a cluster with MS4A1 expression. What cell type is this?
    - Which cell type typically expresses CD8A?
    - What cell type shows high expression of GNLY and NKG7?
    - Cluster Y is enriched with FCGR3A and MS4A7. What cell type is this?
    - What cell type is characterized by FCER1A and CST3 expression?
    - Which cell type typically expresses PPBP?
    
    *Note: Replace X and Y with your cluster numbers, and feel free to combine different markers!*
    """)
    
    # Load model
    model, tokenizer = load_model()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your question about cell types..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_response(prompt, model, tokenizer)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 