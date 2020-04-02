# vpype explorations

My generative/plotter art explorations, packaged as [_vpype_](https://github.com/abey79/vpype) plug-ins.


## Examples

_Section in construction._

### covid

Made with the `msimage` command (Generative Design book's module set). Although it looks random, the background is
based on the SARS-CoV-2 genome.

<img src="examples/covid/1.jpg" width="600px" />

<img src="examples/covid/2.jpg" width="600px" />

<img src="examples/covid/3.jpg" width="300px" />
<img src="examples/covid/4.jpg" width="300px" />


## Installation

See _vpype_'s [installation instructions](https://github.com/abey79/vpype/blob/master/INSTALL.md) for information on how
to install _vpype_.


### Existing _vpype_ installation

Use this method if you have an existing _vpype_ installation (typically in an existing virtual environment) and you
want to make this plug-in available. You must activate your virtual environment beforehand.

```bash
$ pip install git+https://github.com/abey79/vpype-explorations.git#egg=vpype-explorations
```

Check that your install is successful:

```
$ vpype --help
Usage: vpype [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  -v, --verbose
  -I, --include PATH  Load commands from a command file.
  --help              Show this message and exit.

Commands:
[...]
  Plugins:
    alien
[...]
```

### Stand-alone installation

Use this method if you need to edit this project. First, clone the project:

```bash
$ git clone https://github.com/abey79/vpype-explorations.git
$ cd vpype-pixelart
```

Create a virtual environment:

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
```

Install _vpype-explorations_ and its dependencies (including _vpype_):

```bash
$ pip install -e .
```

Check that your install is successful:

```
$ vpype --help
Usage: vpype [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  -v, --verbose
  -I, --include PATH  Load commands from a command file.
  --help              Show this message and exit.

Commands:
[...]
  Plugins:
    alien
[...]
```


## Documentation

The complete plug-in documentation is available directly in the CLI help:

```bash
$ vpype alien --help
```


## License

The code is available under the MIT License. The rest (images, etc.) is CC Attribution-NonCommercial-ShareAlike 4.0.
See the [LICENSE](LICENSE) file for details.