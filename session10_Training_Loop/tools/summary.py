"""Print the headline numbers from results/, for writing the README against."""
import json, pathlib
R = pathlib.Path(__file__).resolve().parent.parent / "results"
def load(n): return json.loads((R / n).read_text())

e1, e2, e3 = load("e1_shapes.json"), load("e2_grad_check.json"), load("e3_accumulation.json")
e4, e5, e6 = load("e4_grad_norm.json"), load("e5_mfu.json"), load("e6_float_bits.json")

print("MACHINE", e5["machine"])
print()
print("E1  params", f"{e1['n_params']:,}", "non-emb", f"{e1['n_params_non_embedding']:,}")
print("E1  activations traced", len(e1["activations"]), "grads match", e1["grad_shapes_match"])
print("E1  training state MiB", round(e1["training_state_mib"],1), "global batch tokens", e1["global_batch_tokens"])
print("E1  micro-batch tokens", e1["micro_batch_tokens"])
print()
b = e2["float64"]["best"]
print("E2  f64 backward", e2["float64"]["backward_reported"], "nudge", b["finite_difference"])
print("E2  f64 decimals", round(b["decimals"],2), "rel", b["rel_error"], "h", b["h"])
print("E2  f32 best decimals", round(e2["float32"]["best"]["decimals"],2))
print("E2  toy", e2["toy_chain"]["by_hand"]["dL_dw1"], e2["toy_chain"]["autograd_w1"], e2["toy_chain"]["finite_difference_w1"])
print()
c = e3["one_step"]["comparison"]
print("E3  angle", round(c["angle_degrees"],2), "relL2", round(100*c["relative_l2_difference"],2),
      "cos", round(c["cosine_similarity"],6), "loss err%", round(c["reported_loss_error_pct"],3))
print("E3  micro tokens", e3["one_step"]["micro_batch_tokens"])
print("E3  gap bucketed", round(e3["gap_bucketed"],4), "fixed", round(e3["gap_fixed"],4),
      "final", {k: round(v,4) for k,v in e3["final_val"]["bucketed"].items()})
print("E3  steps", e3["steps"])
mm = e3["curves"]["bucketed"]["mean_of_means"]
err = [100*(a-b_)/b_ for a,b_ in zip(mm["reported"], mm["honest"])]
print("E3  printed-loss mean err %", round(sum(err)/len(err),3))
print()
print("E4  headline", e4["headline_event"])
print("E4  injected", e4["injected"])
print("E4  n natural", len(e4["natural_events"]), "leads", len(e4["natural_lead_events"]))
off = e4["arms"]["cap off"]
import statistics
print("E4  median norm", round(statistics.median(off["grad_norm"]),3),
      "p99", round(sorted(off["grad_norm"])[int(0.99*len(off["grad_norm"]))],3),
      "at inject", round(off["grad_norm"][e4["inject_at"]],3))
print("E4  median step s", round(statistics.median(off.get("seconds") or [0]),4))
print()
bl = e5["baseline"]
print("E5  peak fp32", round(e5["peak_tflops"],3), "bf16", round(e5["roofline_bf16"].get("best_tflops",0),3))
print("E5  tok/s", round(bl["tokens_per_second"]), "TFLOP/s", round(bl["achieved_tflops"],3),
      "MFU", round(100*bl["mfu"],2))
print("E5  phases", {k: round(100*v,1) for k,v in bl["phase_share"].items()})
print("E5  flops/token", bl["flops_per_token"]["total"], "attn share", round(100*bl["flops_per_token"]["attention_share"],1))
print("E5  widths", [(r["d_model"], round(100*r["mfu"],2)) for r in e5["width_sweep"]])
print("E5  sweeps", [(f"{r['batch_size']}x{r['seq_len']}", round(100*r["mfu"],2)) for r in e5["batch_sweep"]])
print("E5  loader ms", round(1000*bl["loader_seconds_per_step"],1), "step ms", round(1000*bl["seconds"]/bl["timed_steps"],1))
print()
for e in e6["encodings"]:
    print(f"E6  {e['format']:9} {e['bits']:34} {e['hex']:>10} {e['value']:.12f} rel {e['rel_error']:.3e} match {e['matches_hardware']}")
print("E6  updates", e6["real_update_sizes"])
print("E6  stagnation", e6["stagnation"])
