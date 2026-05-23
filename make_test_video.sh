#!/usr/bin/env bash
# Creates a 40-second synthetic test video with spoken audio
set -e

OUT="test_input.mp4"

echo "[1/3] Generating speech audio..."
espeak -s 130 -a 180 -w /tmp/speech.wav \
  "Hey everyone, welcome back to my channel. Today I'm going to walk you through my morning productivity routine. \
   I start every single morning with a big glass of water and then I head straight to my desk. \
   The first thing I do is open my laptop and check my top three priorities for the day. \
   I use a simple notebook to write these down. \
   Having a clean workspace really makes a huge difference to how focused I feel. \
   I also keep a cup of coffee nearby because lets be honest, coffee is life. \
   After about thirty minutes of deep work I take a short walk outside. \
   Fresh air and movement really help me think more clearly. \
   So give this routine a try and let me know in the comments how it works for you. \
   If you found this helpful please hit that like button and subscribe. See you in the next one."

echo "[2/3] Building video track (portrait 1080x1920)..."
# Gradient background that shifts color to simulate different scenes
ffmpeg -y \
  -f lavfi -i "color=c=0x1a1a2e:s=1080x1920:r=30" \
  -f lavfi -i "color=c=0x16213e:s=1080x1920:r=30" \
  -filter_complex "
    [0:v][1:v]blend=all_expr='if(gte(T,20),B,A)':shortest=1[blended];
    [blended]drawtext=fontsize=56:fontcolor=white:x=(w-tw)/2:y=(h-th)/2:
      text='TALKING HEAD VIDEO':enable='between(t,0,10)',
    drawtext=fontsize=46:fontcolor=yellow:x=(w-tw)/2:y=(h/2+80):
      text='Productivity Routine':enable='between(t,0,10)',
    drawtext=fontsize=56:fontcolor=white:x=(w-tw)/2:y=(h-th)/2:
      text='B-ROLL CUT HERE':enable='between(t,10,20)',
    drawtext=fontsize=46:fontcolor=0xff6584:x=(w-tw)/2:y=(h/2+80):
      text='desk / coffee / notebook':enable='between(t,10,20)',
    drawtext=fontsize=56:fontcolor=white:x=(w-tw)/2:y=(h-th)/2:
      text='TALKING HEAD VIDEO':enable='gte(t,20)'
  " \
  -t 40 /tmp/video_only.mp4

echo "[3/3] Merging speech + video..."
# Convert espeak wav to aac-friendly format
ffmpeg -y -i /tmp/speech.wav -ar 44100 -ac 2 /tmp/speech_stereo.wav

ffmpeg -y \
  -i /tmp/video_only.mp4 \
  -i /tmp/speech_stereo.wav \
  -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 128k \
  -shortest "$OUT"

echo ""
echo "Test video created: $OUT"
echo "Resolution: 1080x1920 (portrait 9:16)"
echo "Duration: ~40 seconds"
ls -lh "$OUT"
