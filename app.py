import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PIL import Image
from googletrans import Translator
import requests
import time

st.set_page_config(page_title="Purple AI", page_icon="🟣", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<h1 style='text-align:center;color:#9b59b6;'>🟣 Purple AI</h1>", unsafe_allow_html=True)
    pwd = st.text_input("رمز عبور", type="password")
    if st.button("ورود"):
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("رمز اشتباه!")
    st.stop()

st.markdown("""
<style>
.logo-circle {
    width:80px;height:80px;border-radius:50%;background:#9b59b6;
    display:flex;align-items:center;justify-content:center;
    font-size:40px;color:white;font-weight:bold;font-family:Arial;
    margin:20px auto;box-shadow:0 4px 15px rgba(155,89,182,0.4);
}
</style>
<div class="logo-circle">C</div>
<h1 style='text-align:center;color:#6a1b9a;'>Purple AI</h1>
<hr>""", unsafe_allow_html=True)

menu = st.sidebar.radio("وظیفه", ["💬 چت", "🖼️ عکس", "🌍 ترجمه", "🎥 ویدیو"])

@st.cache_resource
def load_chat():
    tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tok, model

@st.cache_resource
def load_img():
    return pipeline("image-classification", model="microsoft/resnet-50")

ct, cm = load_chat()
ic = load_img()
tr = Translator()

if menu == "💬 چت":
    st.subheader("چت با Purple AI")
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
        st.session_state.ids = None
    u = st.text_input("شما:")
    if st.button("ارسال") and u:
        new_ids = ct.encode(u + ct.eos_token, return_tensors='pt')
        if st.session_state.ids is not None:
            bot_in = torch.cat([st.session_state.ids, new_ids], dim=-1)
        else:
            bot_in = new_ids
        st.session_state.ids = cm.generate(bot_in, max_length=1000, pad_token_id=ct.eos_token_id, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        reply = ct.decode(st.session_state.ids[:, bot_in.shape[-1]:][0], skip_special_tokens=True)
        st.session_state.msgs.append(("👤 شما", u))
        st.session_state.msgs.append(("🤖 Purple AI", reply))
    for s, m in st.session_state.msgs:
        st.markdown(f"**{s}:** {m}")

elif menu == "🖼️ عکس":
    st.subheader("تشخیص عکس")
    up = st.file_uploader("عکس", type=["jpg","png","jpeg"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, use_column_width=True)
        with st.spinner("تحلیل..."):
            res = ic(img)
        for r in res[:3]:
            st.write(f"{r['label']}: {r['score']:.2%}")

elif menu == "🌍 ترجمه":
    st.subheader("ترجمه")
    txt = st.text_area("متن")
    if st.button("ترجمه") and txt:
        det = tr.detect(txt)
        dest = 'en' if det.lang == 'fa' else 'fa'
        result = tr.translate(txt, dest=dest)
        st.success(f"{det.lang} → {dest}: {result.text}")

elif menu == "🎥 ویدیو":
    st.subheader("تولید ویدیو")
    prompt = st.text_input("صحنه (انگلیسی)")
    if st.button("ساخت") and prompt:
        with st.spinner("۳-۵ دقیقه صبر..."):
            r = requests.post(
                "https://api-inference.huggingface.co/models/ali-vilab/modelscope-damo-text-to-video-synthesis",
                json={"inputs": prompt, "options": {"wait_for_model": True}}
            )
            if r.status_code == 200:
                fname = f"video_{int(time.time())}.mp4"
                with open(fname, "wb") as f: f.write(r.content)
                st.video(fname)
                st.download_button("دانلود", open(fname,"rb"), file_name=fname)
            else:
                st.error("خطا")
