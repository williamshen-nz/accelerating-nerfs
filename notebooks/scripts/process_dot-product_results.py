result = {}

for exp_fname in ["/home/workspace/notebooks/designs/dot-product-DUDU/output/timeloop-model.stats.txt",
                    "/home/workspace/notebooks/designs/dot-product-SCDU/output/timeloop-model.stats.txt",
                    "/home/workspace/notebooks/designs/dot-product-SUDU/output/timeloop-model.stats.txt",
                    "/home/workspace/notebooks/designs/dot-product-SUDU/output/no-optimization/timeloop-model.stats.txt",
                    "/home/workspace/notebooks/designs/dot-product-SUDU/output/gating/timeloop-model.stats.txt"]:
    layer_metric_keys = ["energy", "area", "cycle", "gflops", "utilization", "edp"]
    more_layer_metric_keys = ["total topology energy", "total topology area", "max topology cycles"]
    stats_fname = exp_fname
    # process results
    print(exp_fname)
    sub_result = {}
    with open(stats_fname, "r") as fid:
        lines = fid.read().split("\n")[-50:]
        for line in lines:
            line = line.lower()
            for key in layer_metric_keys + more_layer_metric_keys:
                if not line.startswith(key):
                    continue
                metric = line.split(": ")[1].split(" ")[0]
                print(key, eval(metric))
                sub_result[key] = eval(metric)

    # Recompute Energy and Area
    energy_tot = 0
    area_tot = 0
    with open(stats_fname, "r") as fid:
        lines = fid.read().split("\n")
        for line in lines:
            line = line.lower().strip()
            
            if line.startswith("area"):
                metric = line.split(": ")[1].split(" ")[0]
                print("Area", eval(metric), line)
                energy_tot += eval(metric)
            elif line.startswith("energy (total)"):
                metric = line.split(": ")[1].split(" ")[0]
                print("Energy", eval(metric), line)
                area_tot += eval(metric)

    sub_result["total_energy"] = energy_tot
    sub_result["total_area"] = area_tot
    print("Energy", sub_result["total_energy"], "Area", sub_result["total_area"])
    ###


    if len(exp_fname.split('/')) <= 8:
        result[exp_fname.split('/')[5]] =  sub_result
    else:
        result[exp_fname.split('/')[5] + "_" + exp_fname.split('/')[7]] = sub_result

import json
with open('/home/workspace/notebooks/profile_results/dot-product_results.json', 'w') as f:
    json.dump(result, f)