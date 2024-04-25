## Getting Started

Run `pip install -r requirements.txt` to install requisite dependencies. I'm using my own venv instead of anaconda because it was giving me issues. I'd consider it as well if your imports aren't showing up properly. Make a branch and go for it.

## Structure
**data:** holds several preview images that I tested on. Use only 19-1 to 19-4 or fs-1 and fs-3. The rest of the images we are not testing on, and if you need more, grab it from [here](https://www.youtube.com/watch?v=CS7n-n8Hn3E&t=8s).

**split_cells**: this is a dir that will hold many sub-dirs that themselves will contain ALL of the split images are from a given run of `cropper.py`.

**cropper.py**: this is the main file I'm currently using to run everything. It contains methods to: load in a screenshot of a board, detect the corners, crop the image, and write all of the split cells into a sub-dir in `split_cells`. 

It is currently configured to process `fs-1.png` in the data directory, on line 224: `cropped = crop_img("./data/fs1.png")`. Feel free to change this and play around with this. And NOTE, there is no preprocessing being done on the split images currently, so free to add any preprocessing to the returned image before feeding it into `cells = split_into_grid(cropped, pdt, pdb, pdl, pdr)`. More details about the methods:
* `crop_img`, which takes a filepath from the `data` dir, is what does the brunt of the detection, doing all the preprocessing (for detection purposes, not for the CNN). When you run it, the line     `cv2.imshow('corners', img)` shows the detected corners (blue) and general edge (red). Press `0` to close it and move to the next image, and note that if a blue dot for ANY corner is missing, everything else is going to go wrong. The next image shown is the board, cropped to the dimensions of the blue dot.
* `split_into_grid` takes in a cropped image from `crop_img` and splits it into individual cells, and writes it to the `split_cells` dir as a sub-dir. Do the preprocessing between calling this method and the `crop_img` method.

* `filter_close_corners, extend_line, line_intersection,` and `order_points` are all helper methods for corner detection to find the board. It isn't perfect if the board arrangement is weird, but I think it's good enough. 

**sample.py**: I'll rename this later, but this file basically is additional functionality that allows a user to select the region inside of which the board detection will run, in case the video screenshot is too busy. It's not an issue with the current testing video but was in the past and may recur in the future. You can ignore for now.
