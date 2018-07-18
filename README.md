# README

## Setup

Setup environment: 

- python3.6
- install required python libs: ```pip3 install -r requirements.txt```

## Usage

Run the main script in the src folder:

```
python3 subimg.py /path/to/image1.jpg /path/to/image2.jpg
```

To test the main script

- clean up the ```output``` folder
- run the command: ``` python3 test.py ```
- and the script will create an output folder and save test results  

## Discussion

#### Methods

1. template matching (either opencv or skimage)
1. (not implemented) crop images into blocks; send all blocks and template into a CNN (no top layer) to get their vectors; use L2 distance to find their similarity

#### Known Limitations

By the natural of opencv template matching implementation, it can not handle following case well:

1. rescaled or rotated crops
1. jitters: multiple top left for single object
1. similar objects with different colors since the opencv matchTemplate function takes grey image (single channel) as input
