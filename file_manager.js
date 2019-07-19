////////////////////////////
// GLOBALS

var path1 = '/content'        // the cwd of the first file list box
var path2 = '/content'        // the cwd of the second file list box
var selected1 = '/content'    // the path to a file selected in the first file list box
var selected2 = '/content'    // the path to a file selected in the second file list box

// Call python and get a dictionary containing file names and their paths
function get_files(path, list_id) {
  async function foo() {
    console.log(path);
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_cwd_hook', // The callback name.
      [path], // The arguments.
      {}); // kwargs
    return result;
  };
  foo().then(function(value) {
    // parse the return value
    var returned = value.data['application/json'];
    var dict = eval(returned.result);                 // dictionary of filenames and full paths
    var file_list = document.getElementById(list_id); // list box html element
    // Clear the list box
    removeOptions(file_list);                       
    var files = [];             // filenames
    var key;                    
    // Move filesnames (keys) out of dictionary into files list
    for (key in dict) {
      files.push(key);
    }
    // Sort the files
    var sorted_files = files.sort();    // the sorted file list
    // But make sure . is at the top of the list
    var files_temp = []
    files_temp.push('./')
    var i;
    for (i = 0; i < sorted_files.length; i++) {
      if (sorted_files[i] !== './') {
        files_temp.push(sorted_files[i])
      }
    }
    sorted_files = files_temp;
    // ASSERT: files are sorted and . is at the top of the list
    // Populate the list box
    var i;
    for (i = 0; i < sorted_files.length; i++) {
      var file = sorted_files[i];
      var val_path = dict[file];
      var opt = document.createElement('option');
      opt.value = val_path;
      opt.innerHTML = file;
      file_list.appendChild(opt);
    }
  });
  // ASSERT: nothing after here guaranteed to be executed before foo returns
}
  
// Remove all options from a select box
function removeOptions(selectbox) {
    var i;
    for(i = selectbox.options.length - 1 ; i >= 0 ; i--) {
        selectbox.remove(i);
    }
}
  
  
  
// Set up the select boxes with double_click callbacks
var file_list1 = document.getElementById("file_list1");  
file_list1.ondblclick = function(){
  var filename = this.options[this.selectedIndex].innerHTML;
  var path = this.options[this.selectedIndex].value;
  if (filename[filename.length-1] === "/") {
    // this is a directory
    path1 = path;
    selected1 = path;
    update_gui(path, "path1", "file_list1");
  }
};
file_list1.onclick = function() {
  var filename = this.options[this.selectedIndex].innerHTML;
  var path = this.options[this.selectedIndex].value;
  selected1 = path;
};
  
var file_list2 = document.getElementById("file_list2");
file_list2.ondblclick = function(){
  var filename = this.options[this.selectedIndex].innerHTML;
  var path = this.options[this.selectedIndex].value;
  if (filename[filename.length-1] === "/") {
    // this is a directory
    path2 = path;
    selected2 = path;
    update_gui(path, "path2", "file_list2");
  }
};
file_list2.onclick = function() {
  var filename = this.options[this.selectedIndex].innerHTML;
  var path = this.options[this.selectedIndex].value;
  selected2 = path;
};

// update the gui if a path has changed
function update_gui(path, dir_id, list_id) {
  var path_text = document.getElementById(dir_id);
  path_text.innerHTML = path;
  get_files(path, list_id);
}
  
// Copy button
function do_copy_mouse_up() {
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_copy_hook', // The callback name.
      [selected1, selected2], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
  })();
  update_gui(path1, "path1", "file_list1")
  update_gui(path2, "path2", "file_list2")
}
  
// Move button
function do_move_mouse_up() {
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_move_hook', // The callback name.
      [selected1, selected2], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
  })();
  update_gui(path1, "path1", "file_list1")
  update_gui(path2, "path2", "file_list2")
}  
  
// Open text button
function do_open_text_mouse_up() {
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_open_text_hook', // The callback name.
      [selected1], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
  })();
}
  
// Open image button
function do_open_image_mouse_up() {
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_open_image_hook', // The callback name.
      [selected1], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
  })();
}

function do_mkdir_mouse_up() {
  var input_box = document.getElementById('mkdir_input');
  dir_name = input_box.value;
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_mkdir_hook', // The callback name.
      [selected1, dir_name], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
  })();
  update_gui(path1, "path1", "file_list1")
  update_gui(path2, "path2", "file_list2")
}

function do_trash_mouse_up() {
  (async function() {
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_trash_hook', // The callback name.
      [selected1], // The arguments.
      {}); // kwargs
    const res = result.data['application/json'];
  })();
  update_gui(path1, "path1", "file_list1")
  update_gui(path2, "path2", "file_list2")
}
  
// GO
update_gui(path1, "path1", "file_list1")
update_gui(path2, "path2", "file_list2")