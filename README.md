# softscreen-ai
Edge AI based blur filter for TVs using Jetson Nano

---

## What is SoftScreen?

SoftScreen runs a real-time object segmentation and blur engine on a small edge AI device. It watches incoming HDMI content, finds target objects like dogs, nudity, alcohol, or weapons, and applies a natural-looking blur — in real-time.

It’s designed for homes that want peace of mind without censorship, subscriptions, or delay.

Originally built to help dog trainers and reactive pet owners filter animal triggers from TV, it's now evolving into something much bigger.

---

## Powered by Mello

At the core is Mello — our custom detection and blur engine. Mello handles:

- Object segmentation and tracking
- Blur persistence and memory across frames
- Label-based blur prioritization
- Local reflections and session feedback
- Adjustable blur intensity and class targeting

Mello is modular and improving constantly.

---

## Hardware Overview

- Built for Jetson Orin Nano (Model T201)
- HDMI passthrough input/output
- USB 3.0, LAN, and Type-C ports
- Passive cooling, runs fanless
- No cloud or internet required

This is a plug-and-play blur box, built to run quietly alongside your TV.

---

## Who it's for

SoftScreen started for dog owners — especially people dealing with reactivity, anxiety, or distraction. But it quickly became clear this tool helps more than just pets.

This is for:

- Parents who want to filter nudity, violence, or alcohol without turning off the TV
- People with sensory triggers who want more control
- Anyone who wants to reclaim their screen without censorship or judgment

---

## Status

- MVP is live and functional
- Jetson box is fully set up and tested
- Real-time blur confirmed on video tests
- GitHub repo established
- Preparing for small-scale user testing

---

## Want to help?

We're open to feedback, testing partners, and early collaborators. This is a real project solving a real problem. If you're working in edge AI, accessibility, or family tech, we’d love to talk.

Email: softscreenai@gmail.com  
Website: [softscreenai.net](https://softscreenai.net)

---

## Credits

SoftScreen is led by Patrick Wilcox and Maddie Buchholz, with hardware support from Aileen’s team at Twowin.  
Special thanks to Daniel Situnayake for his early encouragement and guidance.

---

The goal is simple: blur what you don’t want to see.  
No cloud. No compromise. Just control.