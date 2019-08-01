function run_program() {
  var input_box = document.getElementById('inp_run');
  var path = input_box.value;
  async function foo() {
    console.log(path);
    const result = await google.colab.kernel.invokeFunction(
      'notebook.python_run_hook', // The callback name.
      [path], // The arguments.
      {}); // kwargs
    return result;
  };
  foo().then(function(value) {});
}