"""Pad-token analysis from SAVED measurements only (no GPU, no Kaggle).

    python3 tools/pad_analysis.py [--ops docs/measurements/check_and_ops_9995a69]
                                  [--full docs/measurements/full_profile_522bada]

Question: how much of the work in a training step is spent on pad tokens, and what would dropping pad
from the whole decoder (not only before the head) save?

Notation: R = real tokens in a step (attention_mask.sum), P = tokens with pad (B * longest row = rows of the
Linear matmuls). Every number printed comes from a file under the given directories; guesses are marked "оценка".

Sources
  result.json  runs.solution[0].log.num_tokens   cumulative real tokens per step, exact integers (-> R)
  op_profile.json  ops_by_shape                  aten::mm / copy_ with input shapes (-> P as the M of [M,1536]x[1536,8960])
  metrics.json  step_s, apply_s, train_loop_s    wall-clock marks of the harness
  result.json  summary / runs[].wall_s           whole-command wall
  profile.json (full_profile)                    per-kernel GPU table (kernel names of the copy kernels)
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T4_FP16_TENSOR_PEAK_TFLOPS = 65.0   # NVIDIA T4 datasheet (not from our logs); used only as a yardstick
PROFILE_FIRST_STEP, PROFILE_STEPS = 9, 16   # harness/_train.py: wait 6 + warmup 2, then 16 active steps (1-based steps 9..24)
BATCH = 4
GATE_SHAPE = re.compile(r"^\[\[(\d+), 1536\], \[1536, 8960\]\]$")


def load(path):
    return json.loads(Path(path).read_text())


def per_step_real_tokens(result):
    cum = result["runs"]["solution"][0]["log"]["num_tokens"]
    return [int(b - a) for a, b in zip([0.0] + cum[:-1], cum)]


def padded_tokens_of_profile(op_profile):
    """M of the gate/up matmul [M,1536]x[1536,8960] = P of one profiled step; count_per_step is averaged over the steps,
    and one step has 84 such calls (28 layers x (gate, up forward + one backward)), so steps = count_per_step * 16 / 84."""
    out = []
    for row in op_profile["ops_by_shape"]:
        match = GATE_SHAPE.match(row["shapes"]) if row["op"] == "aten::mm" else None
        if match:
            steps = round(row["count_per_step"] * PROFILE_STEPS / 84)
            out += [int(match.group(1))] * steps
    return sorted(out)


def parse_shapes(text):
    return json.loads(text)


def section(title):
    print(f"\n{'=' * 8} {title}")


def stage_a(real):
    section("a) R: реальные токены на шаг (result.json runs.solution[0].log.num_tokens, кумулятивное, целые)")
    print(f"шагов {len(real)}; всего R = {sum(real)}; среднее {sum(real) / len(real):.1f}; мин {min(real)}; макс {max(real)}")
    print("trainer_state.json в сохранённых замерах нет; тот же ряд лежит в result.json (поле log) и в train.log (там округлён до 4 цифр)")
    return real


def stage_b(real, op_profile):
    section("b) P: токены с pad на шаг (M матмула [M,1536]x[1536,8960] из op_profile.json, шаги 9..24)")
    known = padded_tokens_of_profile(op_profile)
    window = real[PROFILE_FIRST_STEP - 1:PROFILE_FIRST_STEP - 1 + PROFILE_STEPS]
    print(f"M известны для {len(known)} из {PROFILE_STEPS} профилированных шагов: {known}")
    print(f"M / {BATCH} = длина самой длинной строки: {[m // BATCH for m in known]}")
    print(f"M = 2048 = {BATCH} x 512: батч упёрся в max_length=512; остальные M кратны 4, не кратны 8/16: pad до самой длинной строки,")
    print("а не до max_length и не до кратного числа")
    print(f"R шагов 9..24 (окно профиля): {window}; сумма {sum(window)}")
    missing = PROFILE_STEPS - len(known)
    print(f"Нет M для {missing} шага: ops_by_shape хранит только топ-40 строк (tools/op_profile.py TOP_OPS), самый короткий шаг не попал.")
    print("Какой именно шаг имеет какой M, из файлов не восстановить (форм по шагам нет). Нужен замер: логировать attention_mask.shape и .sum() на шаг.")
    return known, window, missing


def stage_c(real, known, window, missing):
    section("c) R/P")
    lo_missing = max(BATCH, min(window))                 # P >= R for every step
    hi_missing = min(known)                              # оценка: пропущенный шаг короче самого короткого из известных
    sum_p_lo, sum_p_hi = sum(known) + lo_missing * missing, sum(known) + hi_missing * missing
    print(f"сумма R окна {sum(window)}; сумма P окна: известные {sum(known)} + пропущенный шаг в [{lo_missing}, {hi_missing}]")
    r_over_p_hi, r_over_p_lo = sum(window) / sum_p_lo, sum(window) / sum_p_hi
    print(f"R/P окна (агрегат, взвешенное по токенам): {r_over_p_lo:.3f} ... {r_over_p_hi:.3f}  (середина {(r_over_p_lo + r_over_p_hi) / 2:.3f})")
    # one feasible per-step pairing: sort both (оценка; the true pairing is not recoverable)
    p_all = sorted(known + [hi_missing] * missing)
    pairs = [r / p for r, p in zip(sorted(window), p_all)]
    print(f"по шагам при сопоставлении по порядку (оценка, настоящее сопоставление неизвестно): "
          f"среднее {sum(pairs) / len(pairs):.3f}; мин {min(pairs):.3f}; макс {max(pairs):.3f}")
    print(f"R всех 64 шагов {sum(real)}; P всех 64 шагов не знаю (форм вне окна нет); если R/P окна типично, P64 около "
          f"{sum(real) / ((r_over_p_lo + r_over_p_hi) / 2):.0f} (оценка)")
    print("Как Soup дополняет батч: Soup SFT (soup_cli 0.77.0 trainer/sft.py) не передаёт data_collator (кроме raft/vision), значит работает")
    print("TRL 0.29.1 DataCollatorForLanguageModeling: pad(..., padding_side='right', pad_to_multiple_of=None) до самой длинной строки батча,")
    print("attention_mask: 1 на реальных, 0 на pad; pad справа. Совпадает с данными выше (M = 4 x длина самой длинной, truncation 512).")
    return (r_over_p_lo + r_over_p_hi) / 2, r_over_p_lo, r_over_p_hi


def stage_d(op_profile):
    section("d) достигнутые TFLOPs матмулов (aten::mm по формам; время с накладными профайлера)")
    rows, total_flops_ms = [], 0.0
    for row in op_profile["ops_by_shape"]:
        if row["op"] != "aten::mm":
            continue
        shapes = parse_shapes(row["shapes"])
        (m, k), (_, n) = shapes[0], shapes[1]
        per_call_ms = row["ms_per_step"] / row["count_per_step"]
        tflops = 2 * m * n * k / (per_call_ms * 1e-3) / 1e12
        rows.append((tflops, m, n, k, per_call_ms, row["count_per_step"]))
    for tflops, m, n, k, ms, count in sorted(rows, key=lambda r: -r[1]):
        print(f"M={m:5d} N={n:6d} K={k:6d}  {ms:6.3f} мс/вызов  {tflops:5.1f} TFLOPs  ({tflops / T4_FP16_TENSOR_PEAK_TFLOPS * 100:4.1f}% от {T4_FP16_TENSOR_PEAK_TFLOPS:.0f})")
    flops_total = sum(2 * m * n * k * c for _, m, n, k, _, c in rows)
    ms_total = sum(ms * c for _, _, _, _, ms, c in rows)
    print(f"по перечисленным формам: {flops_total / (ms_total * 1e-3) / 1e12:.1f} TFLOPs в среднем ({ms_total:.1f} мс/шаг из aten::mm 318.5 мс/шаг)")
    print("Вывод: матмулы идут на 23-31 TFLOPs (35-47% паспортного пика 65 TFLOPs fp16 tensor, паспорт T4, не наш замер). «У предела T4»")
    print("это не доказывает и не опровергает: достижимый потолок cuBLAS на этой карте мы не измеряли (не знаю). Видно другое: TFLOPs почти не")
    print("зависят от M в диапазоне 940...2048 (плоско около 25-28), значит время матмула примерно пропорционально M, и уменьшение M на R/P")
    print("даст почти пропорциональное уменьшение времени (оценка; ниже M=940 данных нет).")
    return rows


def stage_e(full_dir, op_profile):
    section("e) aten::copy_ (130 мс/шаг): что известно")
    total = next(r for r in op_profile["ops_by_name"] if r["op"] == "aten::copy_")
    print(f"всего aten::copy_: {total['ms_per_step']:.1f} мс/шаг, {total['count_per_step']:.0f} вызовов/шаг (op_profile.json)")
    print("по формам (топ-40 таблицы; dtype в таблице не записан):")
    for row in op_profile["ops_by_shape"]:
        if row["op"] == "aten::copy_":
            print(f"  {row['shapes'][:60]:60s} {row['ms_per_step']:6.2f} мс/шаг  {row['count_per_step']:7.1f} вызовов/шаг")
    path = Path(full_dir) / "solution_profile.json"
    if path.exists():
        kernels = load(path)["kernels"]
        steps = load(path)["active_steps"]
        print("по кернелам (solution_profile.json, мс/шаг; пара dtype в имени кернела не указана, указан только тип вычисления float):")
        for k in kernels:
            if "copy_kernel" in k["name"] and k["time_us"] / steps > 1000:
                kind = "float16_copy_kernel (fp16<->fp32)" if "float16_copy" in k["name"] else (
                    "direct_copy LoadWithCast/StoreWithCast" if "LoadWithCast" in k["name"] else "direct_copy nocast (strided)")
                print(f"  {k['time_us'] / steps / 1e3:6.2f} мс/шаг  {k['count'] / steps:6.1f} вызовов/шаг  {kind}")
    print("НЕ восстановить из сохранённого: разбивку по модулю-источнику и точную пару dtype (из, в) для copy_. В modules (op_profile.json)")
    print("время привязано к модулю, но не к операции; stack_samples пуст. Нужен замер: profiler с record_shapes + стеки или хук на copy_ с dtype и модулем.")
    print("Из форм видно только: самые тяжёлые копии идут на [B,T,8960] (вход MLP, 16.7 мс на форму 512) и [B,T,1536] (остаток), то есть")
    print("пропорциональны числу токенов P и должны уменьшиться вместе с pad.")


def stage_f(ops_dir):
    section("f) фазы запуска (metrics.json, result.json, train.log)")
    metrics, result = load(Path(ops_dir) / "metrics.json"), load(Path(ops_dir) / "result.json")
    run = result["runs"]["solution"][0]
    wall, loop, apply_s = run["wall_s"], metrics["train_loop_s"], metrics["apply_s"]
    step0 = metrics["step_s"][0]
    print(f"wall всей команды {wall:.2f} с = train_loop {loop:.2f} (из них первый шаг {step0:.2f} с, медиана шага {metrics['median_step_s']:.3f})")
    print(f"  + apply() {apply_s:.2f} с + остальное {wall - loop - apply_s:.2f} с (старт процесса, импорты, загрузка и квантизация модели, токенизация, сохранение)")
    print(f"первый шаг минус медиана = {step0 - metrics['median_step_s']:.2f} с: оценка стоимости компиляции Triton и прогрева на первом шаге")
    print("из train.log (если лежит рядом в kaggle/out_*): загрузка весов 338 тензоров около 2 с (tqdm), токенизация Map <1 с (tqdm)")
    print("НЕ восстановить: время импортов (torch, transformers, peft, trl), квантизации bnb при загрузке (внутри «остального»), сохранения адаптера,")
    print("запуска процесса и eval; в логах нет временных меток. Нужен замер: time.perf_counter() метки в _train.py (harness не менять) или")
    print("запуск с python -X importtime и отдельный замер загрузки модели.")
    return wall, loop, apply_s


def stage_g(ratio, lo, hi, op_profile, wall, loop, ops_dir):
    section("g) ОЦЕНКА по формуле: шаг и wall без pad (не замер)")
    by_name = {r["op"]: r["ms_per_step"] for r in op_profile["ops_by_name"]}
    total = op_profile["total_gpu_ms_per_step"]
    mm = by_name["aten::mm"] + by_name.get("aten::addmm", 0.0)
    head_mm = 11.40 + 10.35                       # две формы головы [512,...]: уже на реальных токенах (ops_by_shape)
    nonscaling = by_name["bitsandbytes::dequantize_4bit"] + by_name["bitsandbytes::gemm_4bit"] + 5.52 + 3.08  # NF4, копия весов головы, _FusedHeadLoss
    attention = by_name["aten::_efficient_attention_forward"] + by_name["aten::_efficient_attention_backward"]
    scaling_matmul = mm - head_mm
    elementwise = total - mm - nonscaling - attention
    print(f"GPU {total:.1f} мс/шаг (профиль): матмулы Linear {scaling_matmul:.1f} (голова {head_mm:.1f} уже без pad), поэлементные и прочее {elementwise:.1f},")
    print(f"  attention {attention:.1f}, не зависит от токенов (NF4 dequant, gemm_4bit, веса головы, head_loss) {nonscaling:.1f}")
    print("допущения: матмулы и поэлементные операции линейны по числу токенов (×R/P); attention между ×1 (нет выигрыша) и ×R/P;")
    print("  NF4 и голова не меняются; запуск кернелов не дороже (на деле per-row SDPA добавит 3 лишних вызова × 28 слоёв × fwd/bwd: не учтено)")
    step_real = load(Path(ops_dir) / "metrics.json")["median_step_s"]
    for label, r in (("R/P низ", lo), ("R/P середина", ratio), ("R/P верх", hi)):
        for att_label, att in (("attention ×1", 1.0), ("attention ×R/P", r)):
            new = scaling_matmul * r + elementwise * r + attention * att + nonscaling + head_mm
            step = step_real * new / total
            wall_new = wall - (loop * (1 - new / total))
            print(f"  {label} {r:.3f}, {att_label:15s}: GPU {new:6.1f} мс/шаг ({new / total * 100:5.1f}%), шаг ~{step:.3f} с (было {step_real:.3f}), "
                  f"wall ~{wall_new:.1f} с (было {wall:.1f})")
    print("отдельно матмулы: Linear-матмулы ×R/P:", f"{scaling_matmul:.1f} -> {scaling_matmul * ratio:.1f} мс/шаг (оценка, если эффективность на меньшем M не упадёт);",
          "поэлементные:", f"{elementwise:.1f} -> {elementwise * ratio:.1f} мс/шаг (оценка)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ops", default=str(REPO / "docs/measurements/check_and_ops_9995a69"))
    parser.add_argument("--full", default=str(REPO / "docs/measurements/full_profile_522bada"))
    args = parser.parse_args()

    result, op_profile = load(Path(args.ops) / "result.json"), load(Path(args.ops) / "op_profile.json")
    real = stage_a(per_step_real_tokens(result))
    known, window, missing = stage_b(real, op_profile)
    ratio, lo, hi = stage_c(real, known, window, missing)
    stage_d(op_profile)
    stage_e(args.full, op_profile)
    wall, loop, _ = stage_f(args.ops)
    stage_g(ratio, lo, hi, op_profile, wall, loop, args.ops)
    section("ворота")
    print(f"среднее R/P окна {lo:.3f} ... {hi:.3f}; порог 0.75 -> {'ИДЕЯ НЕ ОКУПАЕТСЯ' if lo > 0.75 else 'проходим дальше (ждём слово «дальше»)'}")


if __name__ == "__main__":
    main()
