#!/usr/bin/env python3
"""Эксп. 25: эволюция геометрических признаков укладки/ротации бедра (DeepSeek как оператор мутации, фитнес — программа).

Цикл: модель пишет функцию extra(lm, img, side, h, w) → fitness.py считает OOF ROC-AUC на поисковой части (60 % исследований)
вместе с боевыми признаками. В элиту попадает кандидат, поднявший AUC хотя бы на MIN_GAIN над текущим лучшим.
В конце три лучших кандидата один раз проверяются на отложенной части (40 % исследований). В бой — только если
ΔAUC на отложенной части ≥ 0,03 и нижняя граница 95 % ДИ > 0. Иначе результат записывается как отрицательный.
    python evolve.py --generations 8 --children 4
"""
import argparse, json, os, random, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import llm_client  # noqa: E402
OUT = os.path.join(ROOT, "data", "work", "exp25"); os.makedirs(OUT, exist_ok=True)
FORBIDDEN = re.compile(r"\b(import|open|exec|eval|compile|__import__|globals|locals|getattr|setattr|delattr|vars|os\.|sys\.|subprocess|"
                       r"__\w+__|np\.(load|save|savez|fromfile|tofile|memmap|genfromtxt|loadtxt|savetxt|lib|ctypeslib|DataSource))\b")
MIN_GAIN = 0.005
SYSTEM = """Ты придумываешь геометрические признаки для оценки укладки и ротации проксимального отдела бедра на денситометрии (DXA, GE Lunar).
Контракт: функция extra(lm, img, side, h, w) -> dict, не более 4 признаков, значения — конечные числа. Доступны только имена np (numpy) и math.
Нельзя import, открывать файлы, exec/eval, обращаться к атрибутам с двойным подчёркиванием. Функция должна работать на любых входах
(точки могут отсутствовать — тогда верни 0 или пропусти признак) и укладываться в 0,2 с на снимок.
Снимок img: 2D float32 0..255, форма (h, w), около 280×260–300 px, 0,65 мм на пиксель; кость светлая, фон тёмный; встречаются чёрные
прямоугольники-маски по краям (их ставит лаборант, это не признак). Снимок приведён к виду ПРАВОГО бедра: таз и седалищная кость справа,
большой вертел слева, диафиз идёт вниз.
lm — словарь точек локализатора {имя: (x, y, уверенность)}, только точки с уверенностью ≥ 0,3; ошибка 3–6 мм:
head — центр головки; gt_top, gt_lat — верхушка и самая наружная точка большого вертела; neck_sup, neck_inf — верхний и нижний контур
шейки в самом узком месте; lt_tip — самая выступающая внутрь точка малого вертела (нет — малый вертел не виден); lt_up, lt_down —
где контур малого вертела переходит в ровный контур кости; ischium — нижняя точка седалищного бугра; shaft_top, shaft_bot — центр
диафиза под малым вертелом и у нижнего края кадра.
Стандарт ISCD: бедро ротировано внутрь на 15–25°, малый вертел виден минимально или не виден; диафиз параллелен оси стола (вертикален);
в кадре головка, шейка, большой вертел, седалищная кость. Признаки неправильной укладки: малый вертел хорошо виден и выступает
(наружная ротация), шейка укорочена в проекции, большой вертел перекрывает шейку, диафиз наклонён, отведение/приведение бедра.
Признаки уже есть в модели: наклон диафиза по массе кости и оценка ансамбля свёрточных сетей (сильная). Нужны признаки, которые несут
НОВУЮ информацию сверх них. Качество — ROC-AUC логистической регрессии по снимкам вне обучения; данных мало (около 90 снимков,
22 с нарушением), поэтому сложные признаки с подгонкой констант переобучаются — предпочитай простые, клинически осмысленные меры."""
IDEAS = [
    "оцени видимость малого вертела: насколько lt_tip выступает за линию между lt_up и lt_down, в долях ширины диафиза",
    "отношение видимой длины шейки (head → середина neck_sup/neck_inf) к диаметру головки или ширине диафиза — укорочение шейки при ротации",
    "профиль яркости по горизонтали через малый вертел: ширина костной полосы на уровне lt_tip против ширины диафиза ниже",
    "шеечно-диафизарный угол по точкам и его отклонение от типичного 125–135°",
    "перекрытие большого вертела и шейки: расстояние gt_lat → neck_sup против ширины шейки",
    "асимметрия контура диафиза: разность расстояний от оси диафиза до медиального и латерального кортикала на нескольких уровнях",
    "положение головки относительно оси диафиза: горизонтальное смещение head от продолжения линии shaft_bot→shaft_top",
    "высота малого вертела над нижним краем шейки (lt_tip.y − neck_inf.y) в долях длины шейки",
    "отведение/приведение: угол линии ischium→head к горизонтали",
    "яркость (средняя) в окне вокруг lt_tip относительно яркости диафиза — проекция малого вертела добавляет кость",
    "упрости: один-два самых надёжных признака лучше четырёх шумных",
    "используй относительные, а не абсолютные размеры — масштаб у снимков немного разный",
]


def ask(parent_code, parent_res, history):
    hint = random.choice(IDEAS)
    hist = "\n".join(f"- признаки {h['features']}: AUC {h['auc']} (база {h['base_auc']})" for h in history[-8:])
    user = (f"Текущий лучший кандидат (AUC на поиске {parent_res['auc']}, база без него {parent_res['base_auc']}):\n```python\n{parent_code}\n```\n"
            f"Недавние попытки:\n{hist or '—'}\nИдея для этой попытки: {hint}.\n"
            "Предложи улучшенную функцию extra целиком. Ответ — один блок ```python ... ``` без пояснений.")
    txt = llm_client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], max_tokens=1800, temperature=0.8)
    return llm_client.extract_code(txt), hint


def score(code, part="search"):
    if FORBIDDEN.search(code) or "def extra" not in code:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "fitness.py"), "--candidate", path, "--part", part],
                           capture_output=True, text=True, timeout=240)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else None
    except (subprocess.TimeoutExpired, ValueError):
        return None
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--generations", type=int, default=8); ap.add_argument("--children", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=0.30); a = ap.parse_args()
    random.seed(25)
    seed = "def extra(lm, img, side, h, w):\n    return {}\n"
    best = {"code": seed, "res": score(seed)}
    log = open(os.path.join(OUT, "evolve_log.jsonl"), "a", encoding="utf-8")
    pool, history = [best], []
    print("база:", best["res"], flush=True)
    for gen in range(a.generations):
        for c in range(a.children):
            if llm_client.cost_usd() > a.budget_usd:
                print("бюджет исчерпан", flush=True); break
            parent = random.choice(pool[-3:]) if random.random() < 0.3 else best
            try:
                code, hint = ask(parent["code"], parent["res"], history)
            except Exception as e:                                       # noqa: BLE001
                print("модель:", e, flush=True); continue
            res = score(code)
            rec = {"gen": gen, "child": c, "hint": hint, "res": res, "code": code, "t": time.strftime("%H:%M:%S"), "usd": llm_client.cost_usd()}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            if not res or res.get("auc") != res.get("auc"):
                print(f"g{gen}c{c}: отказ", flush=True); continue
            history.append(res)
            print(f"g{gen}c{c}: AUC {res['auc']} (база {res['base_auc']}, геометрия без сети {res['geom_alone_auc']}) {res['features']}", flush=True)
            if res["auc"] >= best["res"]["auc"] + MIN_GAIN:
                best = {"code": code, "res": res}; pool.append(best)
                print(f"  ↑ новый лучший {res['auc']}", flush=True)
    # финальная проверка: три лучших различных кандидата — один раз на отложенной части
    cands = sorted([h for h in pool if h["res"]["features"]], key=lambda h: -h["res"]["auc"])[:3]
    final = []
    for h in cands:
        hr = score(h["code"], "holdout")
        final.append({"search": h["res"], "holdout": hr, "code": h["code"]})
        print("отложенная часть:", h["res"]["features"], hr, flush=True)
    passed = [f for f in final if f["holdout"] and f["holdout"]["delta"] >= 0.03 and f["holdout"]["ci"][0] > 0]
    summary = {"base_search": pool[0]["res"], "final": final, "passed": bool(passed), "usd": llm_client.cost_usd(), "calls": llm_client.SPEND["calls"]}
    json.dump(summary, open(os.path.join(OUT, "evolve_summary.json"), "w"), ensure_ascii=False, indent=1)
    print("ИТОГ:", "прошёл ворота" if passed else "ворота не пройдены — в бой ничего не идёт", f"| ${summary['usd']} | обращений {summary['calls']}", flush=True)


if __name__ == "__main__":
    main()
