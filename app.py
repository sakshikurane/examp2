import os
import pickle
import warnings
 
import pandas as pd
from flask import Flask, render_template_string, request
 
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
 
# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netflix.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)   # exact column order the model expects
CLASSES = [str(c) for c in model.classes_]  # Action, Comedy, Documentary, Drama, Horror, Romance, Sci-Fi
 
# --------------------------------------------------------------------------
# Genre display info: emoji, colour, and a few well-known Netflix titles
# --------------------------------------------------------------------------
GENRES = {
    "Action":      {"icon": "💥", "color": "#ff5a36", "titles": ["Extraction", "The Gray Man", "Army of the Dead"]},
    "Comedy":      {"icon": "😂", "color": "#ffc23d", "titles": ["Murder Mystery", "Dolemite Is My Name", "Big Mouth"]},
    "Documentary": {"icon": "🎥", "color": "#3fc1c9", "titles": ["Our Planet", "Making a Murderer", "The Social Dilemma"]},
    "Drama":       {"icon": "🎭", "color": "#a78bfa", "titles": ["The Crown", "Ozark", "Narcos"]},
    "Horror":      {"icon": "👻", "color": "#e11d48", "titles": ["The Haunting of Hill House", "Bird Box", "The Fall of the House of Usher"]},
    "Romance":     {"icon": "💕", "color": "#f472b6", "titles": ["Bridgerton", "Virgin River", "To All the Boys I've Loved Before"]},
    "Sci-Fi":      {"icon": "🚀", "color": "#38bdf8", "titles": ["Stranger Things", "Black Mirror", "Altered Carbon"]},
}
 
# --------------------------------------------------------------------------
# Label-encoded categories (alphabetical, like sklearn's LabelEncoder).
# Position in each list = the number sent to the model. Rename if yours differ.
# --------------------------------------------------------------------------
GENDER   = ["Female", "Male", "Other"]
SUBSCR   = ["Basic", "Premium", "Standard"]
REGION   = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
DEVICE   = ["Desktop", "Laptop", "Mobile", "TV", "Tablet"]
PAYMENT  = ["Credit Card", "Crypto", "Debit Card", "Gift Card", "PayPal"]
YES_NO   = ["No", "Yes"]
FEES     = {0: 8.99, 1: 17.99, 2: 13.99}   # subscription code -> usual monthly fee (Basic / Premium / Standard)
 
 
def opts(lst):
    return list(enumerate(lst))
 
 
# --------------------------------------------------------------------------
# Form definition
# --------------------------------------------------------------------------
SECTIONS = [
    {"title": "👤 Viewer Profile", "fields": [
        {"name": "age", "label": "Age", "kind": "range", "min": 18, "max": 70, "step": 1, "default": 30},
        {"name": "gender", "label": "Gender", "kind": "select", "options": opts(GENDER), "default": 0},
        {"name": "region", "label": "Region", "kind": "select", "options": opts(REGION), "default": 1},
        {"name": "device", "label": "Main Device", "kind": "select", "options": opts(DEVICE), "default": 2},
    ]},
    {"title": "💳 Subscription", "fields": [
        {"name": "subscription_type", "label": "Plan", "kind": "select", "options": opts(SUBSCR), "default": 2},
        {"name": "monthly_fee", "label": "Monthly Fee", "kind": "select",
         "options": [(8.99, "$8.99"), (13.99, "$13.99"), (17.99, "$17.99")], "default": 13.99},
        {"name": "payment_method", "label": "Payment Method", "kind": "select", "options": opts(PAYMENT), "default": 0},
        {"name": "number_of_profiles", "label": "Number of Profiles", "kind": "range",
         "min": 1, "max": 5, "step": 1, "default": 2},
        {"name": "churned", "label": "Cancelled Subscription (churned)", "kind": "select",
         "options": opts(YES_NO), "default": 0},
    ]},
    {"title": "📺 Viewing Behaviour", "fields": [
        {"name": "watch_hours", "label": "Total Watch Hours", "kind": "range",
         "min": 0, "max": 100, "step": 0.5, "default": 40},
        {"name": "avg_watch_time_per_day", "label": "Avg Watch Time per Day", "kind": "range",
         "min": 0, "max": 40, "step": 0.1, "default": 12},
        {"name": "last_login_days", "label": "Days Since Last Login", "kind": "range",
         "min": 0, "max": 60, "step": 1, "default": 10},
    ]},
]
 
 
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def confidence_info(p: float) -> dict:
    """p = top-genre probability (0-1). 7 genres means pure chance is about 14%."""
    if p >= 0.45:
        return {"text": "High", "color": "#46d369"}
    if p >= 0.25:
        return {"text": "Medium", "color": "#ffc23d"}
    return {"text": "Low", "color": "#ff6b6b"}
 
 
# --------------------------------------------------------------------------
# HTML template (Netflix-style dark theme)
# --------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Netflix Genre Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --red:#e50914; --red2:#b20710; --bg:#141414; --card:#1f1f1f; --card2:#262626;
    --line:rgba(255,255,255,.10); --text:#f5f5f1; --muted:#a3a3a3;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Inter',system-ui,sans-serif;color:var(--text);min-height:100vh;padding-bottom:60px;
    background:
      radial-gradient(900px 420px at 50% -80px, rgba(229,9,20,.35), transparent 70%),
      var(--bg);
  }
  .wrap{max-width:1080px;margin:0 auto;padding:0 16px}
  .brand{display:flex;align-items:center;justify-content:space-between;padding:22px 0}
  .logo{font-family:'Bebas Neue',sans-serif;font-size:34px;letter-spacing:.06em;color:var(--red)}
  .tag{font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
  .hero{padding:26px 0 30px;text-align:center}
  .hero h1{font-family:'Bebas Neue',sans-serif;font-weight:400;font-size:clamp(40px,8vw,76px);
    letter-spacing:.03em;line-height:1}
  .hero h1 span{color:var(--red)}
  .hero p{color:var(--muted);max-width:560px;margin:12px auto 0;font-size:15px}
 
  /* result */
  .result{margin-bottom:28px;border-radius:14px;overflow:hidden;background:var(--card);
    border:1px solid color-mix(in srgb,var(--gc) 55%,transparent);
    box-shadow:0 0 50px color-mix(in srgb,var(--gc) 25%,transparent);animation:pop .5s ease}
  @keyframes pop{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
  .banner{display:flex;gap:26px;align-items:center;flex-wrap:wrap;padding:30px;
    background:linear-gradient(110deg,color-mix(in srgb,var(--gc) 38%,#000),#1a1a1a 75%)}
  .poster{width:128px;height:170px;border-radius:10px;display:flex;align-items:center;justify-content:center;
    font-size:60px;background:linear-gradient(160deg,var(--gc),#000);box-shadow:0 10px 30px rgba(0,0,0,.6)}
  .banner .small{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#d4d4d4}
  .banner h2{font-family:'Bebas Neue',sans-serif;font-weight:400;font-size:60px;letter-spacing:.03em;line-height:1.05}
  .conf{display:inline-flex;align-items:center;gap:8px;margin-top:8px;font-size:13px;font-weight:600;
    padding:5px 12px;border-radius:999px;background:rgba(0,0,0,.45);border:1px solid var(--line)}
  .conf i{width:9px;height:9px;border-radius:50%;background:var(--cc)}
  .body{padding:24px 30px 30px;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:28px}
  .sub{font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}
  .bar-row{display:flex;align-items:center;gap:10px;margin:8px 0;font-size:13px}
  .bar-row .n{width:96px;color:#d4d4d4}
  .track{flex:1;height:10px;border-radius:99px;background:#333;overflow:hidden}
  .fill{height:100%;border-radius:99px}
  .bar-row .v{width:50px;text-align:right;font-weight:700}
  .titles{display:grid;gap:10px}
  .title{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:8px;background:var(--card2);
    border:1px solid var(--line);font-size:14px;font-weight:600}
  .title .n{width:26px;height:26px;border-radius:6px;background:var(--red);display:flex;
    align-items:center;justify-content:center;font-size:13px;font-weight:800}
  .note{font-size:12px;color:var(--muted);margin-top:12px;line-height:1.5}
 
  .error{background:rgba(229,9,20,.15);border:1px solid rgba(229,9,20,.6);padding:14px 18px;
    border-radius:10px;margin-bottom:20px}
 
  /* form */
  form{display:grid;gap:20px}
  .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;
    border-top:3px solid var(--red)}
  .card h2{font-size:16px;font-weight:700;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--line)}
  .field{margin-bottom:16px}.field:last-child{margin-bottom:0}
  label{display:flex;justify-content:space-between;font-size:13px;font-weight:500;color:var(--muted);margin-bottom:7px}
  label output{color:#fff;font-weight:700;background:var(--red);padding:0 9px;border-radius:99px}
  select{width:100%;padding:11px 12px;border-radius:8px;font:inherit;color:var(--text);
    background:#2b2b2b;border:1px solid #3d3d3d;outline:none;transition:.2s}
  select option{background:#2b2b2b}
  select:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(229,9,20,.3)}
  input[type=range]{width:100%;-webkit-appearance:none;height:5px;border-radius:99px;background:#3d3d3d;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;
    background:var(--red);border:3px solid #fff;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.6)}
  input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--red);
    border:3px solid #fff;cursor:pointer}
 
  .actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:6px}
  button,.reset{padding:15px 40px;border-radius:6px;border:0;font:inherit;font-weight:700;font-size:16px;
    cursor:pointer;transition:.2s;text-decoration:none;display:inline-block}
  button{color:#fff;background:var(--red);box-shadow:0 8px 26px rgba(229,9,20,.4)}
  button:hover{background:#f6121d;transform:translateY(-2px)}
  .reset{color:var(--text);background:rgba(109,109,110,.5)}
  .reset:hover{background:rgba(109,109,110,.8)}
  footer{text-align:center;color:var(--muted);font-size:12px;margin:34px auto 0;max-width:620px}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand"><span class="logo">GENREFLIX</span><span class="tag">AdaBoost · Viewer Analytics</span></div>
 
  <div class="hero">
    <h1>Predict Your <span>Favourite Genre</span></h1>
    <p>Enter a viewer's profile, subscription and watching habits to see which genre the model picks.</p>
  </div>
 
  {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}
 
  {% if result %}
  <div class="result" style="--gc: {{ result.color }}; --cc: {{ result.conf_color }}">
    <div class="banner">
      <div class="poster">{{ result.icon }}</div>
      <div>
        <div class="small">Predicted favourite genre</div>
        <h2>{{ result.genre }}</h2>
        <span class="conf"><i></i>Model confidence: {{ result.conf_text }} ({{ result.top_pct }}%)</span>
      </div>
    </div>
    <div class="body">
      <div>
        <div class="sub">Genre probabilities</div>
        {% for p in result.probs %}
        <div class="bar-row">
          <span class="n">{{ p.icon }} {{ p.name }}</span>
          <div class="track"><div class="fill" style="width:{{ p.pct }}%;background:{{ p.color }}"></div></div>
          <span class="v">{{ p.pct }}%</span>
        </div>
        {% endfor %}
        <p class="note">With 7 genres, a random guess is about 14% each. When all bars are close together,
          the model has no strong preference.</p>
      </div>
      <div>
        <div class="sub">Popular {{ result.genre }} titles on Netflix</div>
        <div class="titles">
          {% for t in result.titles %}<div class="title"><span class="n">{{ loop.index }}</span>{{ t }}</div>{% endfor %}
        </div>
      </div>
    </div>
  </div>
  {% endif %}
 
  <form method="POST" action="/">
    <div class="grid3">
    {% for s in sections %}
      <div class="card">
        <h2>{{ s.title }}</h2>
        {% for f in s.fields %}
        <div class="field">
          {% set val = values.get(f.name, f.default) %}
          {% if f.kind == 'select' %}
            <label for="{{ f.name }}">{{ f.label }}</label>
            <select id="{{ f.name }}" name="{{ f.name }}">
              {% for code, text in f.options %}
              <option value="{{ code }}" {{ 'selected' if code|string == val|string }}>{{ text }}</option>
              {% endfor %}
            </select>
          {% else %}
            <label for="{{ f.name }}">{{ f.label }}<output id="o_{{ f.name }}">{{ val }}</output></label>
            <input type="range" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}"
                   step="{{ f.step }}" value="{{ val }}"
                   oninput="document.getElementById('o_{{ f.name }}').textContent=this.value">
          {% endif %}
        </div>
        {% endfor %}
      </div>
    {% endfor %}
    </div>
    <div class="actions">
      <button type="submit">▶ Predict Genre</button>
      <a class="reset" href="/">↺ Reset</a>
    </div>
  </form>
  <footer>Built with Flask &amp; scikit-learn. An independent demo project, not affiliated with Netflix.
    Predictions are statistical estimates.</footer>
</div>
 
<script>
  // Keep the monthly fee in step with the chosen plan (Basic / Premium / Standard)
  const FEES = {{ fees|tojson }};
  const plan = document.getElementById('subscription_type');
  const fee = document.getElementById('monthly_fee');
  plan.addEventListener('change', () => {
    const price = FEES[plan.value];
    for (const o of fee.options) if (parseFloat(o.value) === price) fee.value = o.value;
  });
</script>
</body>
</html>
"""
 
 
# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
 
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            v = {name: float(values[name]) for name in FEATURES}
            X = pd.DataFrame([v])[FEATURES]
 
            proba = model.predict_proba(X)[0]
            order = sorted(range(len(CLASSES)), key=lambda i: -proba[i])
            top = CLASSES[order[0]]
            conf = confidence_info(float(proba[order[0]]))
 
            result = {
                "genre": top,
                "icon": GENRES.get(top, {}).get("icon", "🎬"),
                "color": GENRES.get(top, {}).get("color", "#e50914"),
                "titles": GENRES.get(top, {}).get("titles", []),
                "top_pct": round(float(proba[order[0]]) * 100, 1),
                "conf_text": conf["text"],
                "conf_color": conf["color"],
                "probs": [
                    {"name": CLASSES[i], "icon": GENRES.get(CLASSES[i], {}).get("icon", "🎬"),
                     "color": GENRES.get(CLASSES[i], {}).get("color", "#e50914"),
                     "pct": round(float(proba[i]) * 100, 1)}
                    for i in order
                ],
            }
        except (KeyError, ValueError) as e:
            error = f"Please fill every field with valid values ({e})."
        except Exception as e:  # noqa: BLE001
            error = f"Prediction failed: {e}"
 
    return render_template_string(
        TEMPLATE, sections=SECTIONS, values=values, result=result, error=error, fees=FEES
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)
 