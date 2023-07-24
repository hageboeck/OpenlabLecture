# Openlab GPU Lecture Hands on

This repository contains CUDA exercises for CERN Openlab's GPU lecture. There's two methods to work on these exercises:

### Method 1: Ssh / Terminal
- Find a computer with a GPU. At CERN, you can e.g. use
  - `ssh -X lxplus-gpu.cern.ch`
  - `ssh -X lxplus8-gpu.cern.ch`
  - `ssh -X lxplus9-gpu.cern.ch`
  for access to shared GPUs.
- Clone this repo `git clone https://github.com/hageboeck/OpenlabLecture.git`
- `cd OpenlabLecture/source`
- Use a terminal-based editor such as vim, nano, emacs to edit the files or try graphical editors like geany etc if you have an X client on your computer.
- Compiling the executables.
    - Try it manually using `nvcc -O2 -g -std=c++17 <filename>.cu -o <executable>`
    - Use the Makefile, e.g. `make helloWord` for only one executable or `make` to compile all in one go.


### Method 2: GPU-enabled SWAN session (limited number of slots)
- If you don't have a cernbox account yet, go to [cernbox.cern.ch](https://cernbox.cern.ch)
- Once you have a cernbox, click
  [![OpenInSwan](https://swanserver.web.cern.ch/swanserver/images/badge_swan_white_150.png)](https://swan-k8s.cern.ch/user-redirect/download?projurl=https%3A%2F%2Fgithub.com%2Fhageboeck%2FOpenlabLecture.git)
    - Choose the a software stack with CUDA, i.e. "103 Cuda (GPU)"
    - Wait for the container to start up. If it doesn't start up, all GPUs are occupied. You will have to retry later, use method 1, or work in teams.
- Use [FirstSteps notebook](FirstSteps.ipynb) for the first two tasks.
- Finally, go to [Julia notebook](Julia.ipynb).


## 1. Hello world example
Here we have a very basic helloWorld program that prints a message from the host.

Your tasks:
1. Convert the HelloWorld function to a kernel, and call it from main().
1. In the kernel, fill in the variables that print thread index and block index.
1. Try a few launch configurations with more threads / more blocks.

## 2. vectorAdd example
In this example, we add two arrays on the host and on the device. We use timers to measure the execution speed.

Your tasks:
1. Implement an efficient grid-strided loop. Currently, every thread steps through every item.
1. Find an efficient launch configuration to fully use the device.


## 3. Julia example
We compute the Julia and Fatou sets in the complex plane. This requires evaluating a quadratic complex polynomial for more than a million pixels in the complex plane. This is a perfect job for a GPU.

Your tasks:
1. In the `main()` function, allocate memory for all pixels. The writePPM function expects an array of pixels in the form `{y0={x0 x1 x2 x3 ...}, y1={x0 x1 x2 x3 ...}, ... y_n }`, so allocate a one-dimensional array with enough memory for `x*y` pixels. There's already a section that checks for possible cuda errors, so allocate the memory just before. Don't forget to free the memory when you're done.
1. Launch the draft kernel from the main function. Check for possible errors.
1. Figure out a way to compute the pixel indices `i` and `j` from `threadIdx.x` and `blockIdx.x`. Find a kernel launch configuration that covers the entire image.
1. Implement the computation `z = z^2 + c`.
    - We will not use any external complex number classes for this, so square the complex number by hand.
    - Plug the computation in the loop. Check that it runs
        - for a maximum of `maxIter` times
        - or until z starts to diverge (`|z| >= maxMagnitude`).
    - Check that the iteration at which z diverged is recorded in the pixel array. There's already a line of code that should take care of this, but ensure that your iteration counter ends up in that array.
    - Note: We use 256 colours to colour the resulting image. We scale `k` into the range `[1, 256]` for best contrast, but it's not strictly necessary.
1. Check if you can generate a Julia image like this example ![JuliaExample](juliaExample.png)

You can set the real and imaginary part for `c` as command-line arguments:
```
./julia <realPart> <imaginaryPart>
```

Try for example:
- `./julia -0.4 0.6`
- `./julia 0.285 -0.01`

To display the image, you can use imagemagick's `display`:
```
display julia.ppm
```
If `display` doesn't work, check that you have an ssh connection with X forwarding (`ssh -X ...`).


Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
