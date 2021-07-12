
# IshiVision
Computer Vision Based System to read the numbers in an Ishihara Plate test  
### Important note
Under systems like Debian 10 with Cinnamon DE an issues with gtk is present, if that happens to you go to the `common.py` file and add:  
`gi.require_version("Gtk", "2.0")`  
to fix the issue. Any gi/gtk issues can be traced to that missing/extra line depending from your installation.

## Used technology:
- Local maxima method to extract the dominant colors per each plate
- Various OCR algorithm such as:
  - kNN
  - SVM
  - GNB
  - Area matching
  - SSD
  - SAD
  - SIFT

### Documentation
See <file.pdf here> for more informations

## Help:
`-h, --help`              Displays this help  
`-k, --ocr <type>`        Select the type of ocr  
`-t, --train`             Trains the ocr  
`-s, --size <int>`        Selects the size of the train set [default = 2]  
`-l, --load <file>`       Loads the trained file  
`-d, --dump <file>`       Saves the trained data  
`-v, --verbose`           Verbose prints  
`--debug`                 Enables debug features  
`-a, --accuracy <int>`    Calculates the accuracy  
`-c, --char <char>`       Specify the char to test  
`--gkt`                   Enables gkt fixes for debian 10 and OpenCV 3.something  
`--silent`                Produce no output  
`-j <json file>`          Select ocr modules file  
`-p, --show`              Show the images and the internal elaboration passages  

## Available −k parameter 
- `none` - the default test one, will not return a result  
- `sift` - the sift ocr implementation will be run, only -t is supported and -l, -d may not be used.  
- `knn` - the knn ocr will be run, it will require either a data set passed via -l or to generate one via -t.  
- `svm` - the svm ocr will be run, it will require either a data set passed via -l or to generate one via-t.  
- `sksvm` - the sksvm ocr will be run, it will require either a data set passed via -l or to generate onevia -t.  
- `gnb` - the gnb ocr will be run, it will require either a data set passed via -l or to generate one via -t.  
- `area` - the area ocr will be run, only -t is supported and -l, -d may not be used.  
- `sad` - the sad ocr will be run, only -t is supported and -l, -d may not be used.  
- `ssd` - the ssd ocr will be run, only -t is supported and -l, -d may not be used

### Examples
Training a kNN on a 10 sized train set and checking the accuracy with30 generated images for two consecutive times  
`$ python3 test.py -k knn -d data_set -t -s 10 -a 30 --verbose`  
`$ python3 test.py -k knn -l data_set -a 30 --verbose`  
Run to detect the number 0 in a generated ishihara plate with area matching  
`$ python3 test.py -k area -t --char 0 --verbose --show`  
Run to calculate accuracy with 10 images in a generated ishihara plate with area matching  
`$ python3 test.py -k area -t --accuracy 10 --verbose --show`  

# Generator
It's also possible to generate custom Ishiara plates for testing. To do that run `generator.py`.  

## Help
`-h, --help`                  Displays this help  
`-g, --glyph <glyph>`         Render the given glyph  
`bg <comma separated hex>`   Uses the given colors for the background  (e.g "0x000000, 0xAAAAAA")  
`-fg <comma separated hex>`   Uses the given colors for the foreground  
`-v, --verbose`               Verbose prints  
`<file name>`                 Will be saved as a .PNG regardless  
  
### Credits
This program was made for a Computer Vision course in a Master Degree in Computer Engineering by:  
- davidegiordano [davide.giordano6@studio.unibo.it]
- chkrr00k [chkrr00k@cock.li]
