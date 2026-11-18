import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # The second-image epipole is the left null vector of F.
    U, _, _ = np.linalg.svd(F)
    e = U[:, -1]
    return e / e[2] if abs(e[2]) > 1e-12 else e / np.linalg.norm(e)

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    h, w = im.shape[:2]
    T = np.array([[1., 0, -w/2], [0, 1., -h/2], [0, 0, 1.]])
    et = T @ e
    r = np.hypot(et[0], et[1])
    if r < 1e-12:
        raise ValueError('Epipole at image center: this rectification is singular.')
    alpha = 1. if et[0] >= 0 else -1.
    R = np.array([[alpha*et[0]/r, alpha*et[1]/r, 0],
                  [-alpha*et[1]/r, alpha*et[0]/r, 0], [0, 0, 1.]])
    er = R @ et
    G = np.eye(3)
    G[2, 0] = -er[2] / er[0]
    return np.linalg.inv(T) @ G @ R @ T

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)
    x, y, z = e2
    ex = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    M = ex @ F + np.outer(e2, np.ones(3))
    p1 = (H2 @ M @ points1.T).T
    p2 = (H2 @ points2.T).T
    p1 = p1 / p1[:, 2:3]
    p2 = p2 / p2[:, 2:3]
    a = np.linalg.lstsq(p1, p2[:, 0], rcond=None)[0]
    HA = np.eye(3)
    HA[0] = a
    return HA @ H2 @ M, H2

if __name__ == '__main__':
    from pathlib import Path
    import json
    from skimage.transform import ProjectiveTransform, warp
    Path('results').mkdir(exist_ok=True)
    values = {}
    for name in ['set1', 'set2']:
        folder = 'data/' + name
        im1, im2 = [imread(folder + '/image%d.jpg' % i) for i in (1,2)]
        p1, p2 = [get_data_from_txt_file(folder + '/pt_2D_%d.txt' % i) for i in (1,2)]
        F = normalized_eight_point_alg(p1,p2)
        e1, e2 = compute_epipole(F.T), compute_epipole(F)
        H1,H2 = compute_matching_homographies(e2,F,im2,p1,p2)
        def transform(H,p):
            q = (H @ p.T).T
            return q / q[:,2:3]
        q1,q2 = transform(H1,p1),transform(H2,p2)
        # Use one vertical origin for both images so equal rows remain aligned.
        corners=[]
        for im,H in [(im1,H1),(im2,H2)]:
            h,w=im.shape[:2]
            corners.append(transform(H,np.array([[0,0,1],[w-1,0,1],[0,h-1,1],[w-1,h-1,1]])))
        ymin = min(c[:,1].min() for c in corners)
        ymax = max(c[:,1].max() for c in corners)
        fig,axes=plt.subplots(1,2,figsize=(10,6))
        for ax,im,H,c,q in zip(axes,[im1,im2],[H1,H2],corners,[q1,q2]):
            xmin,xmax = c[:,0].min(),c[:,0].max()
            C=np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
            out=warp(im,inverse_map=ProjectiveTransform(C@H).inverse,
                     output_shape=(int(np.ceil(ymax-ymin))+1,int(np.ceil(xmax-xmin))+1),preserve_range=True)
            ax.imshow(np.clip(out,0,255).astype(np.uint8),cmap='gray',vmin=0,vmax=255)
            ax.plot(q[:,0]-xmin,q[:,1]-ymin,'b.',ms=3)
            for y in np.linspace(0,ymax-ymin,12): ax.axhline(y,color='r',lw=.5)
            ax.set_ylim(ymax-ymin,0); ax.set_aspect('auto'); ax.set_axis_off()
        fig.suptitle(name + ': rectified images')
        fig.tight_layout(); fig.savefig('results/'+name+'_rectified.png',dpi=180)
        values[name]=dict(F=F.tolist(),e1=e1.tolist(),e2=e2.tolist(),H1=H1.tolist(),H2=H2.tolist(),
                          vertical_error=float(np.mean(abs(q1[:,1]-q2[:,1]))),first_pair=[q1[0].tolist(),q2[0].tolist()])
        print(name,'\ne1:',e1,'\ne2:',e2,'\nH1:\n',H1,'\nH2:\n',H2,
              '\nMean vertical difference:',values[name]['vertical_error'])
        plt.show()
    Path('results/calculations.json').write_text(json.dumps(values,indent=2))
