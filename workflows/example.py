import numpy as np
from molecule_search_nmr.search import matching_score

## sample data import
[query, pool] = np.load("../resources/sample_data.npz", allow_pickle=True)["data"]  #

## molecule search
results = {}
for mol_id in pool.keys():

    shifts = np.sort(pool[mol_id])
    results[mol_id] = matching_score(query, shifts)


results = dict(sorted(results.items(), reverse=True, key=lambda item: item[1])[:10])
for i, r in enumerate(results):
    print(i + 1, r, results[r])
