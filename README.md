# IshiVision
Computer Vision Based System to read the numbers in an Ishihara Plate test

## Used technology:
- Local maxima method to extract the dominant colors per each plate
- Various OCR algorithm such as:
  - kNN
  - SVM

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

### Credits
This program was made for a Computer Vision course in a Master Degree in Computer Engineering by:  
- davidegiordano [davide.giordano6@studio.unibo.it]
- chkrr00k [chkrr00k@cock.li]
