# make_characteronly_metadata.py

`make_characteronly_metadata.py` aggregates all the information from various CMD databases into a simple and easy to use files that are of the following format 

```
clip_id = {"actor":[], "class":[], "class_confidence":[], "facetrack":[], "facetrack_feature":[]}
```

where each of the fields is a list containing the information about the facetracks



# failed

getfailedtrain.py
simply gets the files for which any form of metadata doesn't exist