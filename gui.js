// GLOBALS ////////////////////////////////////
const screen_width = 2048;      // default screen width
const screen_height = 600;      // default screen height
const module_width = 200;       // default module width
const module_height = 40;       // default module height
const parameter_offset = 10;    // how much to indent parameters
const parameter_width = module_width - (parameter_offset * 2); // default parameter width
const parameter_height = module_height; // default parameter height
const module_spacing = 20;      // how much space between modules
const parameter_spacing = 5;    // how much space between parameters
const cache_path = "cache/"     // cache directory
const image_path = '/nbextensions/google.colab/'; // where to find images
const text_size = 16;           // default text size
const font_size = "" + text_size + "px"; // default font size for html5

// What images to use for different parameter types
var type_images = {"model" : "model.png",
                   "dictionary" : "dictionary.png",
                   "image" : "image.png",
                   "images" : "images.png",
                   "text" : "text.png",
                   "string" : "string.png",
                   "int" : "number.png",
                   "float" : "number.png"};
  
var drag = null; // Thing being dragged
var clicked = null; // Thing that was clicked
var open_param = null; // Which parameter is being edited
var mouseX; // Mouse X
var mouseY; // Mouse Y
var modules = []; // List of modules
var parameters = []; // list of parameters
var connections = []; // List of connections
var clickables = []; // List of clickable things
var dragables = []; // List of dragable things
var module_counters = {}; // Dictionary of counts of each module used
var drag_offset_x = 0; // Keep track off the offset due to scrolling
var drag_offset_y = 0; // Keep track of the offset due to scrolling
var module_counter = 0;    // keep count of number of modules
var parameter_counter = 0; // keep count of number of parameters
  
  
// START GAME /////////////////////////////////
function startGame() {
    // HIDE HTML OBJECTS
    hideInput();
    // POPULATE MODULE MAKING HTML
    populateModuleMaker();
    // START THE GUI RUNNING
    myGameArea.start();
    // EVENT HANDLERS
    // Mouse down
    myGameArea.canvas.onmousedown = function(e) {
       var offset = recursive_offset(myGameArea.canvas)
       var i;
       // Iterate through Dragables 
       for (i = 0; i < dragables.length; i++) {
         var obj = dragables[i];
         if (mouseX >= obj.x - offset.x && mouseX <= obj.x - offset.x + obj.width && 
             mouseY >= obj.y - offset.y && mouseY <= obj.y - offset.y + obj.height) {
            // found the dragable we clicked on
            drag = obj;  // we are dragging this object now
            drag_offset_x = mouseX - obj.x;
            drag_offset_y = mouseY - obj.y;
            // Stop showing the text entry
            hideInput();
            return;
         }
       }
       // Iterate through clickables
       for (i = 0; i < clickables.length; i++) {
         var obj = clickables[i];
         if (mouseX >= obj.x - offset.x && mouseX <= obj.x - offset.x + obj.width && 
             mouseY >= obj.y - offset.y && mouseY <= obj.y - offset.y + obj.height) {
           // found the clickable we clicked on
           if (obj.is_out) {
             // Clicked on an output element that is not connected
             clicked = obj; // we are drawing a line from this object now
             break;
           }
           else if (!obj.is_in && !obj.is_out) {
              // clicked on a non-input connection
              // show input
              var inp = document.getElementById("inp");
              inp.style.display = "block";
              var inp_module = document.getElementById("inp_module");
              inp_module.innerHTML = obj.parent.name;
              var inp_param = document.getElementById("inp_param");
              inp_param.innerHTML = obj.name;
              var val = document.getElementById("inp_val");
              val.value = obj.value;
              open_param = obj;
              return;
           }
         }
       }
       // Make text input invisible
       hideInput();
    }
    // Mouse Move
    myGameArea.canvas.onmousemove = function(e) {
      mouseX = e.clientX;
      mouseY = e.clientY;
    }
    // Mouse up
    myGameArea.canvas.onmouseup = function(e) {
      if (drag) {
        // Done dragging
        drag = null;
      }
      if (clicked) {
        // Done clicking
        var offset = recursive_offset(myGameArea.canvas)
        var i;
        // we are making a line. Where did we drag the line to?
        for (i = 0; i < parameters.length; i++) {
          var obj = parameters[i];
          if (mouseX >= obj.x - offset.x && mouseX <= obj.x - offset.x + obj.width && 
              mouseY >= obj.y - offset.y && mouseY <= obj.y - offset.y + obj.height) {
            // Found the parameter we mouse-uped on
            if (obj.is_in && !obj.connected && obj.parent.name != clicked.parent.name && obj.type === clicked.type) {
              // clicked on an input that isn't already connected
              // Make a connection
              c = new connection(clicked, obj);
              c.id = connections.length;
              connections.push(c);
              obj.connected = true;
              clicked.connected = true;
              var fname = cache_path + obj.type + clicked.id;
              obj.value = fname;
              clicked.value = fname;
            }
          }
        }
        // Not making a line any more
        clicked = null;
      }
    }
    // double click
    myGameArea.canvas.ondblclick = function(e){
      var offset = recursive_offset(myGameArea.canvas)
       var i;
      // Dragables 
       for (i = 0; i < dragables.length; i++) {
         var obj = dragables[i];
         if (mouseX >= obj.x - offset.x && mouseX <= obj.x - offset.x + obj.width && 
             mouseY >= obj.y - offset.y && mouseY <= obj.y - offset.y + obj.height) {
           // found the dragable we double-clicked on.
           // Collapse or un-collapse it
           obj.collapsed = !obj.collapsed;
           break;
         }
       }
    };
    // Key Press
    document.addEventListener('keydown', function(e){
      var offset = recursive_offset(myGameArea.canvas)
      if (e.keyCode == 8 || e.keyCode == 46) {
        // DELETE
        var i;
        // Did we delete a dragable?
        for (i = 0; i < dragables.length; i++) {
          var obj = dragables[i];
          if (mouseX >= obj.x - offset.x && mouseX <= obj.x - offset.x + obj.width && 
              mouseY >= obj.y - offset.y && mouseY <= obj.y - offset.y + obj.height) {
            // Found the dragable we were mousing over when the delete button was pressed
            delete_module(obj);
            break;
          }
        }
      }
    } );
}

// MYGAMEAREA //////////////////////////////////////////
var myGameArea = {
    canvas : document.createElement("canvas"),
    start : function() {
        this.canvas.width = screen_width;
        this.canvas.height = screen_height;
        this.context = this.canvas.getContext("2d");
        document.body.insertBefore(this.canvas, document.body.childNodes[0]);
        this.frameNo = 0;
        this.interval = setInterval(updateGameArea, 20);
        },
    clear : function() {
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}


// MODULE ///////////////////////////////////////////////
function module(x, y, type, name, category, id) {
    this.width = module_width;    // module width
    this.height = module_height;  // module height
    this.color = "gray";          // module color
    this.category = category;     // what category did this belong in (not sure this is needed)
    this.x = x;                   // x location
    this.y = y;                   // y location
    this.name = name;             // label to display to user
    this.type = type;             // module type
    this.params = [];             // list of parameters
    this.font = "Arial"           // font to use
    this.font_size = font_size;   // font size
    this.id = id;               // unique id number
    this.collapsed = false;       // am I collapsed?
    this.up = new Image();        // up arrow
    this.up.src = image_path + "up.png"; // load the up arrow image
    this.down = new Image();      // down arrow
    this.down.src = image_path + "down.png"; // load the down arrow image
    // UPDATE FUNCTION
    this.update = function() {
        ctx = myGameArea.context;
        if (drag == this) {
          // I AM BEING DRAGGED
          this.x = mouseX - drag_offset_x;
          this.y = mouseY - drag_offset_y;
        }
        // DRAW ME
        // draw my parameters
        var i;
        for (i = 0; i < this.params.length; i++) {
          param = this.params[i];
          var index = i + 1;
          if (this.collapsed) {
            index = 0;
          }
          param.x = this.x + parameter_offset;
          param.y = this.y + (this.height + parameter_spacing)*(index);
          param.update();
        }
        // Draw the module header
        ctx.fillStyle = this.color;
        ctx.fillRect(this.x, this.y, this.width, this.height);
        ctx.font = this.font_size + " " + this.font;
        ctx.fillStyle = "black";
        var text_x = this.x + 5;
        var text_y = this.y + (this.height / 2.0) + (text_size / 3.0);
        var img_height = (this.height - 10) / 4;
        var img_width = (this.height - 10) / 2;
        ctx.fillText(this.name, text_x, text_y);
        // Show up arrow or down arrow?
        if (this.collapsed) {
          ctx.drawImage(this.down, this.x + this.width - img_width - 5, this.y + 5, img_width, img_height);
        }
        else {
          ctx.drawImage(this.up, this.x + this.width - img_width - 5, this.y + 5, img_width, img_height);
        }
        
    }
}
  
// PARAMETER //////////////////////////
function parameter(name, is_in, is_out, type, default_value, parent, id) {
    this.color = "lightgray";          // parameter color (red is output, green is input, lightgray otherwise)
    this.width = parameter_width;      // parameter width
    this.height = parameter_height;    // parameter height
    this.is_in = is_in;                // am I an input?
    this.is_out = is_out;              // am I an output?
    this.type = type;                  // what type am I? (string, text, int, float, dictionary, model, etc.)
    this.value = default_value;        // my value
    this.x = 0;                        // x location (parent module will set me)
    this.y = 0;                        // y location (parent module will set me)
    this.name = name;                  // My label to show the user
    this.connected = false;            // Am I connected to another parameter?
    this.parent = parent;              // Who is my parent module?
    this.font = "Arial"                // My font
    this.font_size = font_size         // my font size
    this.id = id;                    // unique identifier
    // Set my color based on whether I am an input or output. Also set the default filename I create when linked
    if (this.is_in) {
      this.color = "green";
      this.value = cache_path + name;
    }
    else if (this.is_out) {
      this.color = "red";
      this.value = cache_path + name;
    }
    // Do I have an icon to show?
    this.img = null; // icon image
    if (this.type in type_images) {
      this.img = new Image();
      this.img.src = image_path + type_images[this.type];
    }
    // UPDATE FUNCTION
    this.update = function() {
        ctx = myGameArea.context;
        // DRAW ME
        ctx.fillStyle = this.color;
        ctx.fillRect(this.x, this.y, this.width, this.height);
        ctx.font = this.font_size + " " + this.font;
        ctx.fillStyle = "black";
        var text_x = this.x + 5;
        var text_y = this.y + (this.height / 2.0) + (text_size / 3.0);
        var img_height = this.height - 10;
        var img_width = img_height;
        // Show my icon
        if (this.is_out) {
          ctx.drawImage(this.img, this.x + this.width - img_width - 5, this.y + 5, img_width, img_height);
        }
        else if (this.is_in || this.img) {
          ctx.drawImage(this.img, text_x, this.y + 5, img_width, img_height);
          text_x = text_x + img_width + 5;  
        }
        ctx.fillText(this.name, text_x, text_y);
        
    }
}
  
// CONNECTION ////////////////////////////////
function connection (origin, target) {
  this.origin = origin;        // my origin parameter
  this.target = target;        // my target parameter
  this.id = null;              // unique identifier
  this.update = function() {
      var ctx = myGameArea.context;
      // DRAW ME
      var origin_x = this.origin.x + this.origin.width;
      var origin_y = this.origin.y + this.origin.height/2;
      var target_x = this.target.x;
      var target_y = this.target.y + this.target.height/2;
      if (this.origin.parent.collapsed) {
        origin_x = this.origin.x + this.origin.width/2;
        origin_y = this.origin.y + this.target.height;
      }
      if (this.target.parent.collapsed) {
        target_x = this.target.x + this.target.width/2;
        target_y = this.target.y;
      }
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(origin_x, origin_y);
      ctx.lineTo(target_x, target_y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(origin_x, origin_y, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(target_x, target_y, 5, 0, 2 * Math.PI);
      ctx.fill();
  }
}

// UPDATE CANVAS ///////////////////////////////////
function updateGameArea() {
    var x, height, gap, minHeight, maxHeight, minGap, maxGap;
    myGameArea.clear();
    myGameArea.frameNo += 1;
    // Draw my modules
    var i;
    for (i = 0; i < modules.length; i++) {
      modules[i].update();
    }
    // Draw connections
    var j;
    for (j = 0; j < connections.length; j++) {
      connections[j].update();
    }
    // Is there a line being drawn that hasn't been connected yet?
    if (clicked) {
      var offset = recursive_offset(myGameArea.canvas)
      var ctx = myGameArea.context;
      var dot_x = clicked.x + clicked.width;
      var dot_y = clicked.y + clicked.height/2;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(dot_x, dot_y);
      ctx.lineTo(mouseX+offset.x, mouseY+offset.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(clicked.x + clicked.width, clicked.y + clicked.height/2, 5, 0, 2 * Math.PI);
      ctx.fill();
    }

}
  
// BUTTON HANDLERS /////////////////////////////
  
function do_input_button_up () {
  if (open_param != null) {
    var val = document.getElementById("inp_val");
    open_param.value = val.value;
  }
  hideInput();
}

function do_make_module_button_up() {
  var sel = document.getElementById("module_select");
  var val = sel.options[sel.selectedIndex].value;  // Name of the module type
  var i;
  // Figure out what type of module was selected
  for (i = 0; i < module_dicts.length; i++) {
    var module_dict = module_dicts[i];
    if ("name" in module_dict) {
      var name = module_dict["name"];
      if (name == val) {
        // Found it
        make_module(module_dict, module_counter);
      }
    }
  }
}

  
// HELPERS ///////////////////////////////////////
  
  
function make_module(module_json, id) {
  var category = "";
  var name = "";
  var type = "";
  var module_count = 0;
  // type
  if ("name" in module_json) {
    type = module_json["name"];
  }
  // name
  if (!(type in module_counters)) {
    module_counters[type] = 0;
  }
  module_count = module_counters[type] + 1;
  module_counters[type] = module_count;
  name = type + " (" + module_count + ")";
  // category
  if ("category" in module_json) {
    category = module_json["category"];
  }
  // make new module
  var new_module = new module(((module_width+module_spacing) * modules.length) % (screen_width-(module_width+module_spacing)), 
                              parseInt((module_width+module_spacing) * modules.length / (screen_width-(module_width+module_spacing))) * (module_width+module_spacing), 
                              type, name, category, id);
  module_counter = module_counter + 1;
  //new_module.id = module_counter;
  modules.push(new_module);
  dragables.push(new_module);
  // parameters
  if ("params" in module_json) {
    var params = module_json["params"];
    var j;
    for (j = 0;  j < params.length; j ++ ) {
      param_json = params[j];
      var p_name = ""
      var is_in = false;
      var is_out = false;
      var type = "";
      var default_value = "";
      // parameter name/type
      if ("name" in param_json) {
        p_name = param_json["name"];
      }
      // is it an input?
      if ("in" in param_json && param_json["in"] == true) {
        is_in = true;
      }
      // is it an output?
      if ("out" in param_json && param_json["out"] == true) {
        is_out = true;
      }
      // What type of value does it store?
      if ("type" in param_json) {
        type = param_json["type"];
      }
      // What is the default value?
      if ("default" in param_json) {
        default_value = param_json["default"];
      }
      // Make the parameter
      new_param = new parameter(p_name, is_in, is_out, type, default_value, new_module, new_module.id + "-" + j);
      //new_param.id = new_module.id + "-" + j;
      parameter_counter = parameter_counter + 1;
      new_module.params.push(new_param);
      parameters.push(new_param);
      if (!is_in) {
        clickables.push(new_param);
      }
    }
  }
  return new_module;
}  
  
function hideInput() {
  var inp = document.getElementById("inp");
  inp.style.display = "none";
}
  
function populateModuleMaker() {
  // Grab the select object
  var sel = document.getElementById("module_select");
  var categories_dict = {}; // keys are category names, val is list of module names
  // seed with misc category
  categories_dict["misc"] = [];
  // Collect up category names and module names
  var i;
  for (i = 0; i < module_dicts.length; i++) {
    var module_dict = module_dicts[i];
    if ("name" in module_dict && "category" in module_dict) {
      var category = module_dict["category"];
      var name = module_dict["name"];
      if (!(category in categories_dict)) {
          categories_dict[category] = [];
      }
      categories_dict[category].push(name);
    }
    else if ("name" in module_dict) {
      var name = module_dict["name"];
      categories_dict["misc"].push(name);
    }
  }
  // Iterate through categories
  for (var key in categories_dict) {
    var module_names = categories_dict[key];
    // If this category has modules, then add them to the select object
    if (module_names.length > 0) {
      // This category has modules
      // Make an optgroup
      var group = document.createElement('OPTGROUP');
      group.label = key;
      // Iterate through modules
      for (i=0; i < module_names.length; i++) {
        // Make an option
        var module_name = module_names[i];
        var opt = document.createElement('OPTION');
        opt.textContent = module_name;
        opt.value = module_name;
        group.appendChild(opt);
      }
      sel.appendChild(group);
    }
  }
}
  
function everyinterval(n) {
    if ((myGameArea.frameNo / n) % 1 == 0) {return true;}
    return false;
}


function recursive_offset (aobj) {
 var currOffset = {
   x: 0,
   y: 0
 } 
 var newOffset = {
     x: 0,
     y: 0
 }    

 if (aobj !== null) {

   if (aobj.scrollLeft) { 
     currOffset.x = aobj.scrollLeft;
   }

   if (aobj.scrollTop) { 
     currOffset.y = aobj.scrollTop;
   } 

   if (aobj.offsetLeft) { 
     currOffset.x -= aobj.offsetLeft;
   }

   if (aobj.offsetTop) { 
     currOffset.y -= aobj.offsetTop;
   }

   if (aobj.parentNode !== undefined) { 
      newOffset = recursive_offset(aobj.parentNode);   
   }

   currOffset.x = currOffset.x + newOffset.x;
   currOffset.y = currOffset.y + newOffset.y; 
 }
 return currOffset;
}
  
  
function save_program() {
  // Don't save if there is a blank program.
  if (modules.length == 0) {
    return;
  }
  var filename = "myprogram"
  var input_save = document.getElementById("inp_save");
  if (input_save.value.length > 0) {
    filename = input_save.value;
  }
  var program = [];
  // Deep copy connections
  var module_links = [];
  var module_ids = [];
  var i;
  for (i = 0; i < modules.length; i++) {
    module_ids.push(parseInt(modules[i].id));
  }
  for (i = 0; i < connections.length; i++) {
    module_links.push([parseInt(connections[i].origin.parent.id), parseInt(connections[i].target.parent.id)]);
  }
  // Put all the modules in order
  while (module_ids.length > 0) {
    var ready = get_ready_modules(module_ids, module_links);
    program = program.concat(ready);
    var filtered_module_ids = module_ids.filter(function(value, index, arr) {
      var n;
      var is_in = false;
      for (n=0; n < ready.length; n++) {
        if (value == ready[n]) {
          is_in = true;
          break;
        }
      }
      return !is_in;
    });
    var filtered_module_links = module_links.filter(function(value, index, arr) {
      var n;
      var is_in = false;
      for (n=0; n < ready.length; n++) {
        if (value[0] == ready[n] || value[1] == ready[n]) {
          is_in = true;
          break;
        }
      }
      return !is_in;
    });
    module_ids = filtered_module_ids;
    module_links = filtered_module_links;
  }
  // assert: program is module ids in executable order
  var prog_json = "";
  var is_first = true;
  console.log(program);
  for (i = 0; i < program.length; i++) {
    var current = program[i];
    var current_module = get_module_by_id(current)
    var module_json = module_to_json(current_module);
    if (is_first) {
      prog_json = module_json;
    }
    else {
      prog_json = prog_json + "," + module_json;
    }
    console.log(prog_json);
    is_first = false;
  }
  // Put brackets around the json
  prog_json = "[" + prog_json + "]";
  // Call python
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_save_hook', // The callback name.
      [prog_json, filename], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
    //document.querySelector("#output-area").appendChild(document.createTextNode(text.result));
  })();
}
  
function get_module_by_id(id) {
  var i = 0;
  for (i=0; i < modules.length; i++) {
    if (modules[i].id == id) {
      return modules[i];
    }
  }
  return null;
}
 
  
function get_ready_modules(mods, cons) {
  var ready = [];
  var i;
  for (i=0; i < mods.length; i++) {
    var current = mods[i];
    var is_ready = true;
    var j;
    for (j=0; j < cons.length; j++) {
      if (cons[j][1] == current) {
        is_ready = false;
        break;
      }
    }
    if (is_ready) {
      ready.push(current);
    }
  }  
  return ready;
}
  
function module_to_json(module) {
  var json = {};
  json["module"] = module.type;
  json["name"] = module.name;
  json["x"] = module.x;
  json["y"] = module.y;
  json["id"] = module.id;
  json["collapsed"] = module.collapsed;
  // Parameters
  var i=0;
  for (i=0; i < module.params.length; i++) {
    param = module.params[i];
    json[param.name] = ""+param.value; // Make all values strings
  }
  return JSON.stringify(json);
}

function clear_program() {
  modules = [];
  parameters = [];
  connections = [];
  clickables = [];
  dragables = [];
  module_counters = {};
  module_counter = 0;
  parameter_counter = 0;
  drag_offset_x = 0;
  drag_offset_y = 0;
  drag = null;
  clicked = null;
  open_param = null;
}  
  
function load_program(loadfile="") {
  // Need to clear the existing program
  clear_program();
  var filename = loadfile;
  if (filename.length == 0) {
      // Filename wasn't passed in so we need to get the filename from the html
      var input_load = document.getElementById("inp_load");
      if (input_load.value.length > 0) {
        filename = input_load.value;
      }
      else {
        return;
      }
  }
  // ASSERT: filename is non-empty
  // Call python
  async function foo() {
    const result = await google.colab.kernel.invokeFunction(
        'notebook.python_load_hook', // The callback name.
        [filename], // The arguments.
        {}); // kwargs
    //program = result.data['application/json'];
    //res = result.data['application/json'];
    //console.log(result);
    //document.querySelector("#output-area").appendChild(document.createTextNode(text.result));
  return result;
  };
  foo().then(function(value) {
    var program = eval(value.data['application/json'].result);
    console.log(program);
    // what to do with the program
    var i;
    var m;
    // Iterate through each module in the program
    for (m = 0; m < program.length; m++) {
      var module_json = program[m]; // The json for this part of the program
      var mod = null;
      var id = module_counter; 
      if ("id" in module_json) {
        id = module_json["id"];
      }
      // Find the corresponding module definition
      for (i = 0; i < module_dicts.length; i++) {
        var module_dict = module_dicts[i];
        
        if ("name" in module_dict) {
          var name = module_dict["name"];
          if (name == module_json.module) {
            // Make the module
            mod = make_module(module_dict, id);
            // Except it only has default values
            break;
          }
        }
      } // end i
      // Move the module to the saved location
      mod.x = module_json.x;
      mod.y = module_json.y;
      mod.collapsed = module_json.collapsed;
      // Update default parameters
      // Iterate through each of the specs from the file
      var module_keys = Object.keys(module_json);
      for (i = 0; i < module_keys.length; i++) {
        var module_key = module_keys[i];
        var module_val = module_json[module_key];
        var p;
        // Find the corresponding parameter 
        for (p = 0; p < mod.params.length; p++) {
          var param = mod.params[p];
          if (module_key === param.name) {
            // Found the right parameter
            param.value = module_val;
            break;
          }
        }
      }
    }  //end m
    // Make connections
    console.log(parameters);
    var i;
    var j;
    // Iterate through all parameters, looking for matching file names
    for (i = 0; i < parameters.length; i++) {
      for (j = 0; j < parameters.length; j++) {
        var param1 = parameters[i];
        var param2 = parameters[j];
        if (param1.is_out && param2.is_in) {
          // param1 has an outlink and param2 has an inlink
          var fname1 = param1.value;
          var fname2 = param2.value;
          if (fname1 === fname2) {
            // Make a connection
            c = new connection(param1, param2);
            c.id = connections.length;
            connections.push(c);
            param1.connected = true;
            param2.connected = true;
          }
        }
      }
    }
    // update module count to be max id
    var mod;
    var max_id = 0;
    for (mod = 0; mod < modules.length; mod++) {
      cur_id = modules[mod].id;
      if (cur_id > max_id) {
        max_id = cur_id;
      }
    }
    module_counter = max_id+1;
  }); // end then function
  // Anything after this is not guaranteed to execute after the file is loaded.
}
  
function delete_module(module) {
  var filtered_modules = modules.filter(function(value, index, arr) {
    return value.id != module.id;
  });
  var filtered_connections = connections.filter(function(value, index, arr) {
    return value.origin.parent.id != module.id && value.target.parent.id != module.id;
  });
  var deleted_connections = connections.filter(function(value, index, err) {
    return value.origin.parent.id == module.id || value.target.parent.id == module.id;
  });
  var i;
  for (i = 0; i < deleted_connections.length; i++) {
    var c = deleted_connections[i];
    c.origin.connected = false;
    c.target.connected = false;
  }
  modules = filtered_modules;
  connections = filtered_connections;
}

// START THE GUI ///////////////////////////////////////////
startGame()