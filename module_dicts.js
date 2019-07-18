var module_dicts = [{"name" : "ReadWikipedia", 
                 "params" : [{"name" : "wiki_directory", "type" : "directory", "default" : "wiki"},
                             {"name": "pattern", "type" : "string", "default": "*"}, 
                             {"name" : "categories", "type" : "string", "default" : "*"}, 
                             {"name" : "out_file", "type" : "text", "out" : true}, 
                             {"name" : "titles_file", "type" : "text", "out" : true}], 
                 "category" : "Wikipedia"},
                {"name" : "WordRNN_Train", 
                 "params" : [{"name" : "data", "in" : true, "type" : "text"},
                             {"name" : "history", "type" : "int", "default" : 35},
                             {"name" : "layers", "type" : "int", "default": 2},
                             {"name" : "hidden_nodes", "type" : "int", "default" : 512},
                             {"name" : "epochs", "type" : "int", "default" : 50},
                             {"name" : "learning_rate", "type" : "float", "default": 0.0001},
                             {"name" : "model", "type" : "model", "out" : true},
                             {"name" : "dictionary", "type" : "dictionary", "out" : true}],
                 "category" : "RNN"},
                {"name" : "WordRNN_Run",
                 "params" : [{"name" : "model", "in" : true, "type" : "model"},
                             {"name" : "dictionary", "in" : true, "type" : "dictionary"},
                             {"name" : "seed", "in" : true, "type" : "text"},
                             {"name" : "steps", "type" : "int", "default" : "600"},
                             {"name" : "temperature", "type" : "float", "default" : "0.5"},
                             {"name" : "k", "type" : "int", "default" : "40"},
                             {"name" : "output", "out" : true, "type" : "text"}],
                 "category" : "RNN"},
                {"name" : "MakeString",
                 "params" : [{"name" : "string", "type" : "string"},
                             {"name" : "output", "out" : true, "type" : "text"}],
                 "category" : "Input"},
                {"name" : "RemoveEmptyLines", 
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "SplitSentences",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "ReplaceCharacters",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "find", "type" : "string"},
                             {"name" : "replace", "type" : "string"},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "ReadTextFile",
                 "params" : [{"name" : "file", "type" : "directory"},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "File"},
                {"name" : "WriteTextFile",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "file", "type" : "directory"}],
                 "category" : "File"},
                {"name" : "MakeLowercase",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "Wordify",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "RemoveTags",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "HTML"},
                {"name" : "CleanText",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "SaveModel",
                 "params" : [{"name" : "model", "in" : true, "type" : "model"},
                             {"name" : "file", "type" : "directory"}],
                 "category" : "File"},
                {"name" : "LoadModel",
                 "params" : [{"name" : "file", "type" : "directory"},
                             {"name" : "model", "out" : true, "type" : "model"}],
                 "category" : "File"},
                {"name" : "SaveDictionary",
                 "params" : [{"name" : "dictionary", "in" : true, "type" : "dictionary"},
                             {"name" : "file", "type" : "directory"}],
                 "category" : "File"},
                {"name" : "LoadDictionary",
                 "params" : [{"name" : "file", "type" : "directory"},
                             {"name" : "dictionary", "out" : true, "type" : "dictionary"}],
                 "category" : "File"},
                {"name" : "SplitHTML",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "HTML"},
                {"name" : "RandomSequence",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                  			 {"name" : "length", "type" : "int", "default" : 100},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "ConcatenateTextFiles",
                 "params" : [{"name" : "input_1", "type" : "text", "in" : true},
                             {"name" : "input_2", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "RandomizeLines",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "KeepFirstLine",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "DeleteFirstLine",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "DeleteLastLine",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "KeepLineWhen",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "match", "type" : "string"},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "KeepLineUnless",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "match", "type" : "string"},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "Sort",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "Reverse",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "GPT2_FineTune",
                 "params" : [{"name" : "model_in", "type" : "model", "in" : true},
                 			 {"name" : "data", "type" : "text", "in" : true},
                 			 {"name" : "model_size", "type" : "string", "default" : "117M"},
                 			 {"name" : "steps", "type" : "int", "default" : 1000},
                             {"name" : "model_out", "type" : "model", "out" : true}],
                 "category" : "GPT2"},
                {"name" : "GPT2_Run",
                 "params" : [{"name" : "model_in", "type" : "model", "in" : true},
                 			 {"name" : "prompt", "type" : "text", "in" : true},
                 			 {"name" : "model_size", "type" : "string", "default" : "117M"},
                             {"name" : "top_k", "type" : "int", "default" : 40},
                             {"name" : "temperature", "type" : "float", "default" : 1.0},
                             {"name" : "num_samples", "type" : "int", "default" : 1},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "GPT2"},
                {"name" : "CharRNN_Train",
                 "params" : [{"name" : "data", "in" : true, "type" : "text"},
                             {"name" : "history", "type" : "int", "default" : 35},
                             {"name" : "layers", "type" : "int", "default": 2},
                             {"name" : "hidden_nodes", "type" : "int", "default" : 512},
                             {"name" : "epochs", "type" : "int", "default" : 50},
                             {"name" : "learning_rate", "type" : "float", "default": 0.0001},
                             {"name" : "model", "type" : "model", "out" : true},
                             {"name" : "dictionary", "type" : "dictionary", "out" : true}],
                 "category" : "RNN"},
                {"name" : "CharRNN_Run",
            	 "params" : [{"name" : "model", "in" : true, "type" : "model"},
                             {"name" : "dictionary", "in" : true, "type" : "dictionary"},
                             {"name" : "seed", "in" : true, "type" : "text"},
                             {"name" : "steps", "type" : "int", "default" : "600"},
                             {"name" : "temperature", "type" : "float", "default" : "0.5"},
                             {"name" : "output", "out" : true, "type" : "text"}],
                 "category" : "RNN"},
                {"name" : "UserInput",
                 "params" : [{"name" : "prompt", "type" : "string", "default" : "prompt"},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Input"},
                {"name" : "Regex_Search",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "expression", "type" : "string", "default" : "*"},
                             {"name" : "output", "type" : "text", "out" : true},
                             {"name" : "group_1", "type" : "text", "out" : true},
                             {"name" : "group_2", "type" : "text", "out" : true}],
                 "category" : "Regex"},
                {"name" : "Regex_Sub",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "expression", "type" : "string", "default" : ""},
                             {"name" : "replacement", "type" : "string", "default" : ""},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Regex"},
                {"name" : "PrintText",
                 "params" : [{"name" : "input", "type" : "text", "in" : true}],
                 "category" : "Utils"},
                {"name": "ReadFromWeb",
                 "params" : [{"name" : "url", "type" : "string", "default" : ""},
                             {"name" : "data", "type" : "text", "out" : true}],
                 "category" : "Web"},
                {"name" : "MakeCountFile",
                 "params" : [{"name" : "num", "type" : "int", "default" : "10"},
                             {"name" : "prefix", "type" : "string", "default" : ""},
                             {"name" : "postfix", "type" : "string", "default" : ""},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Utils"},
                {"name" : "ReadAllFromWeb",
                 "params" : [{"name" : "urls", "type" : "text", "in" : true},
                             {"name" : "data", "type" : "text", "out" : true}],
                 "category" : "Web"},
                {"name" : "RemoveDuplicates",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "StripLines",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "TextSubtract",
                 "params" : [{"name" : "main", "type" : "text", "in" : true},
                             {"name" : "subtract", "type" : "text", "in" : true},
                             {"name" : "diff", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "DuplicateText",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "count", "type" : "int", "default" : 1},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "Spellcheck",
                 "params" : [{"name" : "input", "type" : "text", "in" : true},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Text"},
                {"name" : "WebCrawl",
                 "params" : [{"name" : "url", "type" : "string", "default" : ""},
                             {"name" : "link_id", "type" : "string", "default" : ""},
                             {"name" : "link_text", "type" : "string", "default" : ""},
                             {"name" : "max_hops", "type" : "int", "default" : "10"},
                             {"name" : "output", "type" : "text", "out" : true}],
                 "category" : "Web"},
                {"name" : "ScrapePinterest",
                 "params" : [{"name" : "url", "type" : "string", "default" : ""},
                             {"name" : "username", "type" : "string", "default" : ""},
                             {"name" : "password", "type" : "string", "default" : ""},
                             {"name" : "target", "type" : "int", "default" : "100"},
                             {"name" : "output", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "LoadImages",
                 "params" : [{"name" : "directory", "type" : "directory", "default" : ""},
                             {"name" : "images", "type" : "images", "out" : true}],
                 "category" : "Files"},
                {"name" : "SaveImages",
                 "params" : [{"name" : "images", "type" : "images", "in" : true},
                             {"name" : "directory", "type" : "directory", "default" : ""}],
                 "category" : "Files"},
                {"name" : "ResizeImages",
                 "params" : [{"name" : "input", "type" : "images", "in" : true},
                             {"name" : "size", "type" : "int", "default" : "256"},
                             {"name" : "output", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "RemoveGrayscale",
                 "params" : [{"name" : "input", "type" : "images", "in" : true},
                             {"name" : "output", "type" : "images", "out" : true},
                             {"name" : "rejects", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "CropFaces",
                 "params" : [{"name" : "input", "type" : "images", "in" : true},
                             {"name" : "size", "type" : "int", "default" : "256"},
                             {"name" : "output", "type" : "images", "out" : true},
                             {"name" : "rejects", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "StyleGAN_FineTune",
                 "params" : [{"name" : "model_in", "type" : "model", "in" : true},
                             {"name" : "images", "type" : "images", "in" : true},
                             {"name" : "start_kimg", "type" : "int", "default" : "7000"},
                             {"name" : "max_kimg", "type" : "int", "default" : "25000"},
                             {"name" : "seed", "type" : "int", "default" : "1000"}, 
                             {"name" : "schedule", "type" : "string", "default" : ""},                            
                             {"name" : "model_out", "type" : "model", "out" : true},
                             {"name" : "animation", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "StyleGAN_Run",
                 "params" : [{"name" : "model", "type" : "model", "in" : true},
                             {"name" : "num", "type" : "int", "default" : "1"},
                             {"name" : "images", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "StyleGAN_Movie",
                 "params" : [{"name" : "model", "type" : "model", "in" : true},
                             {"name" : "length", "type" : "int", "default" : "10"},
                             {"name" : "interp", "type" : "int", "default" : "10"},
                             {"name" : "duration", "type" : "int", "default" : "10"},
                             {"name" : "movie", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "MakeMovie",
                 "params" : [{"name" : "images", "type" : "images", "in" : true},
                             {"name" : "duration", "type" : "int", "default" : "10"},
                             {"name" : "movie", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "Gridify",
                 "params" : [{"name" : "input", "type" : "images", "in" : true},
                             {"name" : "size", "type" : "int", "default" : "256"},
                             {"name" : "columns", "type" : "int", "default" : "4"},
                             {"name" : "output", "type" : "images", "out" : true}],
                 "category" : "Images"}, 
                {"name" : "Degridify",
                 "params" : [{"name" : "input", "type" : "images", "in" : true},
                             {"name" : "columns", "type" : "int", "default" : "4"},
                             {"name" : "rows", "type" : "int", "default" : "4"},
                             {"name" : "output", "type" : "images", "out" : true}],
                 "category" : "Images"},
                {"name" : "StyleTransfer",
                 "params" : [{"name" : "content_image", "type" : "images", "in" : true},
                             {"name" : "style_image", "type" : "images", "in" : true},
                             {"name" : "steps", "type" : "int", "default" : "1000"},
                             {"name" : "size", "type" : "int", "default" : "512"},
                             {"name" : "style_weight", "type" : "int", "default" : "1000000"},
                             {"name" : "content_weight", "type" : "int", "default" : "1"},
                             {"name" : "content_layers", "type" : "string", "default" : "4"},
                             {"name" : "style_layers", "type" : "string", "default" : "1, 2, 3, 4, 5"},
                             {"name" : "output", "type" : "images", "out" : true}],
                 "category" : "Images"}
               ];
/*

    {"name" : "MakePredictionData", "params" : "data(in,text);x(out,text);y(out,text)", "tip" : "Prepare data for prediction--each line will try to predict the next line", "category" : "no"}, \
    {"name" : "DCGAN_Train", "params" : "input_images(images,in);epochs(int=10);input_height(int=108);output_height(int=108);filetype(string=jpg);model(out,model);animation(out,image)", "tip" : "Train a generateive adversarial network to make images", "category" : "no"}, \
    {"name" : "DCGAN_Run", "params" : "input_images(images,in);model(in,model);input_height(int=108);output_height(int=108);filetype(string=jpg);output_image(out,image)", "tip" : "Generate an image from a generative adversarial network", "category" : "no"}, \
    {"name" : "ReadImages", "params" : "data_directory(directory);output_images(out,images)", "tip" : "Read in a directory of image files", "category" : "File"}, \
    {"name" : "WriteImages", "params" : "input_images(in,images);output_directory(directory)", "tip" : "Save a group of images to a directory", "category" : "File"}, \
    {"name" : "PickFromWikipedia", "params" : "wiki_directory(directory,tip=Directory where wikipedia files are stored);input(in,text);catgories(string=*,tip=What categories if any?);section_name(string,tip=What section to pull text from if any);output(out,text);break_sentences(bool=false,tip=Should text be broken into one sentence per line?)", "tip" : "Pull text from wikipedia for the articles specified (file with one title per line)", "category" : "Wikipedia"}, \
    {"name" : "Repeat", "params" : "input(in,text);output(out,text);times(int)", "category" : "Do not use"}, \
    {"name" : "StyleNet_Train", "params" : "style_image(in,image);test_image(in,image);epochs(int=2,tip=how long to run);model(out,model);animation(out,image)", "tip" : "Draw the target image in the style of the style image", "category" : "no"}, \
    {"name" : "StyleNet_Run", "params" : "model(in,model);target_image(in,image,tip=Image to stylize);output_image(out,image)", "tip" : "Apply a style learned by a neural net to an image", "category" : "no"}, \
    {"name" : "ReadImageFile", "params" : "file(string,tip=Name of image file to read in);output(out,image)", "tip" : "Read an image file in", "category" : "File"}, \
    {"name" : "WriteImageFile", "params" : "input(in,image);file(string,tip=Name of image file to write to)", "tip" : "Write an image to file", "category" : "File"}, \
    {"name" : "MakeEmptyText", "params" : "output(out,text)", "tip" : "Create an empty text file", "category" : "Utils"}, \
]';
*/