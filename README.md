# Handy

Budget JARVIS, for a wannabe Iron Man.

## Testing

Below are the steps to get the program set up

```shell
git clone https://github.com/kevinyhe/handy
cd handy
py -m venv venv
venv\scripts\activate.bat
py -3 -m pip install -r requirements.txt
```

## Running

Run `py app.py` in your shell, and click start webcam. For reference here is a list of gestures:

Thumb - index: Left Click
Thumb - middle: Right Click
Index - middle: Move
Middle - ring: Scroll
Thumb - index, and ring - pinky curled downwards: Drag

## Inspiration

Really a silly source of inspiration, but more or less I was inspired to make this after the left click buttons on my mouse broke, and I was left to deal with a trackpad for over a week. This led to irreversible (negligible) pain both physically and emotionally; therefore, I had to replace it with hopefully, at least in my eyes, something more convenient.

## What it does

Handy takes your generic webcam and turns it into a mouse sensor - you can now use your hand as if you had a dollar store mouse duct taped to it! With Handy, you can achieve all basic functionality, including clicking, scrolling, dragging, and obviously moving. We've also designed the controls to be as intuitive as possible to make it as user-friendly as possible.

## How we built it

We built Handy with python - we used opencv to get camera data and used mediapipe to locate where each point of interest on the hand was. We also used PyQt5 for the GUI and interface.
