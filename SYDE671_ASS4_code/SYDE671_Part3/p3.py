import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    structure, motion, _ = factorization_details(points_im1, points_im2)
    return structure, motion

def factorization_details(points_im1, points_im2):
    from scipy.optimize import minimize_scalar
    W = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T))
    W = W - W.mean(axis=1, keepdims=True)
    U, sigma, Vt = np.linalg.svd(W, full_matrices=False)
    root = np.sqrt(sigma[:3])
    Mhat = U[:, :3] * root
    Shat = root[:, None] * Vt[:3]
    def sym(a, b):
        return [a[0]*b[0], a[1]*b[1], a[2]*b[2],
                a[0]*b[1]+a[1]*b[0], a[0]*b[2]+a[2]*b[0],
                a[1]*b[2]+a[2]*b[1]]
    def matrix(l):
        return np.array([[l[0],l[3],l[4]], [l[3],l[1],l[5]],
                         [l[4],l[5],l[2]]])
    A, b = [], []
    for k in (0, 2):
        a, c = Mhat[k:k+2]
        A.extend([sym(a,a), sym(c,c), sym(a,c)])
        b.extend([1.,1.,0.])
    A, b = np.array(A), np.array(b)
    l = np.linalg.lstsq(A, b, rcond=None)[0]
    _, sa, va = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    # Two orthographic views generally leave one metric degree of freedom.
    # Choose a positive-definite representative by maximizing its minimum
    # eigenvalue along the null direction. This is a gauge choice, NOT GT fitting.
    if rank == 5:
        null = va[-1]
        bound = 100 * max(np.linalg.norm(l), 1.)
        opt = minimize_scalar(lambda t: -np.linalg.eigvalsh(matrix(l+t*null))[0],
                              bounds=(-bound,bound), method='bounded')
        l = l + opt.x*null
    L = matrix(l)
    eig, V = np.linalg.eigh(L)
    floor = max(np.max(np.abs(eig))*1e-10, 1e-12)
    projected = bool(np.any(eig <= floor))
    Q = V @ np.diag(np.sqrt(np.maximum(eig, floor)))
    motion = Mhat @ Q
    structure = np.linalg.solve(Q, Shat)
    details = dict(singular_values=sigma, constraint_singular_values=sa,
                   constraint_rank=rank, metric_eigenvalues=eig,
                   spd_projected=projected, metric_residual=np.linalg.norm(A@l-b),
                   W=W, affine_structure=Shat,
                   reconstruction_rmse=np.sqrt(np.mean((W-motion@structure)**2)))
    return structure, motion, details

if __name__ == '__main__':
    from pathlib import Path
    import json
    Path('results').mkdir(exist_ok=True)
    results={}
    for name in ['set1','set1_subset']:
        folder='data/'+name
        p1,p2=[get_data_from_txt_file(folder+'/pt_2D_%d.txt'%i) for i in (1,2)]
        gt=get_data_from_txt_file(folder+'/pt_3D.txt')[:,:3]
        S,M,d=factorization_details(p1,p2)
        # Similarity alignment is only for display; GT is never used to estimate S.
        X=S.T-S.T.mean(axis=0); Y=gt-gt.mean(axis=0)
        u,sv,vt=np.linalg.svd(X.T@Y)
        R=u@vt; scale=sv.sum()/np.sum(X*X)
        aligned=scale*X@R
        fig=plt.figure(figsize=(10,5))
        lim=np.max(np.abs(Y))*1.1
        for i,(pts,title) in enumerate([(aligned,'Recovered (similarity aligned)'),(Y,'Ground truth')]):
            ax=fig.add_subplot(1,2,i+1,projection='3d')
            ax.scatter(*pts.T,c=np.arange(len(pts)),cmap='viridis',s=22)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim),xlabel='X',ylabel='Y',zlabel='Z',title=title)
            ax.set_box_aspect((1,1,1)); ax.view_init(20,-65)
        fig.suptitle(name); fig.tight_layout()
        fig.savefig('results/'+name+'_structure.png',dpi=180)
        d.update(motion=M,structure=S,point_count=len(gt),gt_singular_values=np.linalg.svd(Y,compute_uv=False),
                 aligned_relative_error=np.linalg.norm(aligned-Y)/np.linalg.norm(Y))
        results[name]={k:v.tolist() if isinstance(v,(np.ndarray,np.generic)) else v for k,v in d.items()}
        print(name,'\nSingular values:',d['singular_values'],'\nMetric constraint rank:',d['constraint_rank'],
              '\nReconstruction RMSE:',d['reconstruction_rmse'])
        plt.show()
    Path('results/calculations.json').write_text(json.dumps(results,indent=2))
