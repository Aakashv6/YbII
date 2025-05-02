import lib.peaks as peaks
import numpy as np
import matplotlib.pyplot as plt
import lib.plotting.pyplotdefs as pd
import cv2


img = np.array(plt.imread("data/pictures/array20x20.tiff"))
# img = np.load("data/pictures/final20x20.npy")
# peak_locs = np.load("data/peaks_full.npy")
w,h = (60,60)
powers = np.array([
    img[j0 - h // 2 : j0 + h // 2, i0 - w // 2 : i0 + w // 2].sum()
    for i0, j0 in peak_locs
]).reshape(5,5)

# (pd.Plotter()
#     .imshow(img, cmap='gray')
#     .set_xticklabels([])
#     .set_yticklabels([])
#     # .colorbar()
#     .grid(False)
#     .savefig("data/pictures/final20x20.pdf", dpi=1200)
#     .close()
# )
peak_finder = peaks.Cv2HoughCircles(
    gaussian_blur_params={"ksize": (5,5),"sigmaX": 0.0,"sigmaY": 0.0},
    dp=1.5,
    method=cv2.HOUGH_GRADIENT,
    minDist=70,
    param1=40,
    param2=8,
    minRadius=4,
    maxRadius=8,
)

peak_locs, proc = peak_finder.find_peaks(img)
peak_locs.sort(key=lambda xy: xy[1])
# peak_locs[37][0] -= 8
# peak_locs[37][1] -= 6
# x = (peak_locs[30][0] + peak_locs[37][0]) / 2
# y = (peak_locs[30][1] + peak_locs[37][1]) / 2
# peak_locs.append((int(x),int(y)))
# peak_locs.sort(key=lambda xy: xy[1])
# peak_locs[39][0] += 10
# peak_locs[39][1] -= 16
# peak_locs[115][0] -= 15
# peak_locs[115][1] -= 15
# peak_locs[123][0] += 10
# peak_locs[123][1] += 5
# peak_locs
print(f"raw locs: {len(peak_locs)}")


sort_shape = (5, 5)
group_dist = 300.
appending_list = []
sorted_rows = []
for i, (x,y) in enumerate(peak_locs):
    appended = False
    for j, row in enumerate(appending_list):
        if abs(row[-1][0] - x) < group_dist:
            row.append((x,y))
            appended = True
            if len(row) == sort_shape[1]:
                sorted_rows.append(row)
                appending_list.pop(j)
    if not appended:
        appending_list.append([])
        appending_list[-1].append((x,y))
    
grouped_peak_locs = [xy for row in sorted_rows for xy in row]
# grouped_peak_locs[0], grouped_peak_locs[-1] = grouped_peak_locs[-1], grouped_peak_locs[0]
print(f"sorted: {len(grouped_peak_locs)}")

for k, (x,y) in enumerate(peak_locs):
    cv2.putText(
        img,
        str(k),
        (x-10, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
    

(pd.Plotter()
    .imshow(img, cmap="jet")
    .grid(False)
    .set_xticklabels([])
    .set_yticklabels([])
    # .colorbar()
    .savefig("figs/peaks.png", dpi=1200)
    .close()
)

np.save("data/peaks_full.npy", grouped_peak_locs)