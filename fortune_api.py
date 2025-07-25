from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
from functools import lru_cache
import threading

app = Flask(__name__)

# 1. 경량 공개 모델 로딩 (인증 불필요한 t5-small)
generator = pipeline(
    "text2text-generation",
    model="t5-small"
)

# 2. 동일 프롬프트 결과 캐싱 (최대 128개)
@lru_cache(maxsize=128)
def generate_cached(prompt: str) -> str:
    result_container = {"text": ""}

    def worker():
        try:
            out = generator(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
            # t5-small은 'generated_text' 키 사용
            result_container["text"] = out[0].get("generated_text", "").strip()
        except Exception as e:
            result_container["text"] = f"⚠️ 생성 오류: {e}"

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=15)  # 15초 이상 지연 시 포기

    text = result_container["text"]
    return text or "운세 생성에 문제가 발생했어요. 다시 시도해 주세요."

@app.route("/", methods=["GET"])
def home():
    return "✅ 운세 생성 API입니다. POST 요청은 /fortune 경로로 보내주세요."

@app.route("/fortune", methods=["POST"])
def generate_fortune():
    try:
        data     = request.get_json(force=True)
        name     = data.get("name", "")
        birth    = data.get("birth", "")
        time     = data.get("time", "")
        calendar = data.get("calendar", "")
        gender   = data.get("gender", "")
        category = data.get("category", "건강운")
        typ      = data.get("type", "오늘")

        # 3. 프롬프트 템플릿
        templates = {
            "건강운": f"{name}님의 생년월일은 {birth}, 태어난 시간은 {time}, 성별은 {gender}, {calendar} 기준이에요. {typ}의 건강운을 알려주세요. 생활 습관, 컨디션, 조심할 점을 중심으로 2~3줄의 짧은 조언을 친구처럼 따뜻하게 말해줘요.",
            "재물운": f"{name}님의 생년월일은 {birth}, 태어난 시간은 {time}, 성별은 {gender}, {calendar} 기준이에요. {typ}의 재물운을 알려주세요. 돈의 흐름과 주의할 점을 중심으로 친근한 말투로 조언해주세요.",
            "연애운": f"{name}님의 생년월일은 {birth}, 태어난 시간은 {time}, 성별은 {gender}, {calendar} 기준이에요. {typ}의 연애운을 알려주세요. 감정, 인연, 관계를 중심으로 따뜻한 말투로 말해주세요.",
            "학업운": f"{name}님의 생년월일은 {birth}, 태어난 시간은 {time}, 성별은 {gender}, {calendar} 기준이에요. {typ}의 학업운을 알려주세요. 집중력, 공부운, 성취를 중심으로 2~3줄 조언을 부탁해요.",
            "취업운": f"{name}님의 생년월일은 {birth}, 태어난 시간은 {time}, 성별은 {gender}, {calendar} 기준이에요. {typ}의 취업운을 알려주세요. 기회, 경쟁, 자신감을 중심으로 말해주세요."
        }

        prompt = templates.get(category, templates["건강운"])
        app.logger.debug(f"📝 Prompt: {prompt}")

        # 4. 캐시된 생성 호출
        result = generate_cached(prompt)
        app.logger.debug(f"📥 Result: {result}")

        return jsonify({"prompt": prompt, "result": result})

    except Exception as e:
        app.logger.error(f"Exception in /fortune: {e}", exc_info=True)
        return jsonify({"prompt": "", "result": f"서버 오류: {e}"}), 500

@app.route("/test", methods=["GET"])
def test_page():
    return render_template_string("""
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>운세 API 테스트</title>
</head>
<body style="font-family: sans-serif; padding:20px;">
  <h2>운세 생성 API 테스트</h2>
  <form id="f">
    이름: <input name="name" value="재우"><br>
    생년월일: <input name="birth" value="2002-01-01"><br>
    시간: <input name="time" value="오전 9시"><br>
    양/음력: <input name="calendar" value="양력"><br>
    성별: <input name="gender" value="남자"><br>
    카테고리:
      <select name="category">
        <option>건강운</option>
        <option>재물운</option>
        <option>연애운</option>
        <option>학업운</option>
        <option>취업운</option>
      </select><br>
    타입:
      <select name="type">
        <option>오늘</option>
        <option>내일</option>
      </select><br><br>
    <button type="submit">발송</button>
  </form>
  <div id="status" style="margin:8px 0; color:#555;">준비 완료</div>
  <pre id="out" style="background:#f0f0f0;padding:10px;"></pre>

  <script>
    const statusEl = document.getElementById('status');
    const outEl    = document.getElementById('out');
    document.getElementById('f').onsubmit = async e => {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(e.target).entries());
      statusEl.textContent = '⏳ 요청 중...';
      outEl.textContent = '';
      try {
        const res = await fetch('/fortune', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        outEl.textContent =
          '📝 프롬프트:\\n' + json.prompt +
          '\\n\\n📥 결과:\\n' + json.result;
        statusEl.textContent = '✅ 완료';
      } catch (err) {
        statusEl.textContent = '❌ 에러 발생';
        outEl.textContent = err.message;
        console.error(err);
      }
    };
  </script>
</body>
</html>
""")

if __name__ == "__main__":
    app.run(debug=True)
