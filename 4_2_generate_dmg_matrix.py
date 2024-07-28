import numpy as np
from utils.parse_args import parse_args, show_args
import tqdm
import multiprocessing as mp
import scipy
import sklearn
from sklearn.neighbors import NearestNeighbors


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over


adp_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Adp{r}Scene{s}.npy"
dmg_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Dmg{r}Scene{s}.npy"

n_neighbors = 20
adp = np.load(adp_path, mmap_mode="r")

adp = np.clip(adp, 0, None)

print(np.min(adp))

nbrs_alg = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", n_jobs=-1)
nbrs = nbrs_alg.fit(adp)
nbg = sklearn.neighbors.kneighbors_graph(
    nbrs, n_neighbors, metric="precomputed", mode="distance"
)


dissimilarity_matrix_geodesic = np.zeros((nbg.shape[0], nbg.shape[1]), dtype=np.float32)


def shortest_path_worker(todo_queue, output_queue):
    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        d = scipy.sparse.csgraph.dijkstra(nbg, directed=False, indices=index)
        output_queue.put((index, d))


thread_num = 12
with tqdm.tqdm(total=nbg.shape[0] ** 2) as pbar:
    todo_queue = mp.Queue()
    output_queue = mp.Queue()

    for i in range(nbg.shape[0]):
        todo_queue.put(i)
    for i in range(thread_num):
        todo_queue.put(-1)
        p = mp.Process(target=shortest_path_worker, args=(todo_queue, output_queue))
        p.start()

    finished_processes = 0
    while finished_processes != thread_num:
        i, d = output_queue.get()

        if i == -1:
            finished_processes = finished_processes + 1
        else:
            dissimilarity_matrix_geodesic[i, :] = d
            pbar.update(len(d))

np.save(dmg_path, dissimilarity_matrix_geodesic)
