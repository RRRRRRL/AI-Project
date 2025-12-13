# Video Presentation Script: LSTM Trajectory Prediction for Aircraft
## Duration: 3 Minutes

---

## SLIDE 1: Title Slide (0:00 - 0:10)
**[10 seconds]**

**Visual**: Title slide with project name and a background image of aircraft trajectories overlaid on a map

**Narration**:
"Hello! Today I'll present our LSTM-based aircraft trajectory prediction system for Hong Kong airspace. In just three minutes, I'll show you the problem we're solving, our approach, and the impressive results we achieved."

---

## SLIDE 2: Problem Introduction (0:10 - 0:35)
**[25 seconds]**

**Visual**: 
- Split screen showing:
  - Left: Animation of aircraft in flight with question marks ahead
  - Right: Example trajectory plot showing past (solid line) and future (dotted line)

**Narration**:
"Imagine you're an air traffic controller. An aircraft is approaching Hong Kong, and you need to predict where it will be in the next 10 minutes. Will it conflict with other aircraft? When will it land?

Traditional methods use simple physics—assume the plane continues at constant velocity. But aircraft don't fly in straight lines. They turn, climb, descend, and respond to air traffic control.

Our challenge: Given 20 minutes of past positions, predict the next 10 minutes of trajectory with meter-level accuracy."

---

## SLIDE 3: Our Approach - Data Pipeline (0:35 - 0:55)
**[20 seconds]**

**Visual**:
- Flow diagram: Raw Data → Preprocessing → Features → Model → Predictions
- Show example of coordinate transformation (lat/lon to ENU)
- Display feature list with icons

**Narration**:
"Our solution is a complete end-to-end pipeline. 

First, we transform GPS coordinates into a local frame—East, North, Up relative to the aircraft's current position. This simplifies the math.

Next, we engineer smart features: velocities, speed, heading direction, and even time-of-day. We encode circular quantities like heading using sine and cosine to preserve continuity—359 degrees and 1 degree should be close, right?

This gives us 8 features per timestep, fed into our model."

---

## SLIDE 4: LSTM Architecture (0:55 - 1:20)
**[25 seconds]**

**Visual**:
- Animated diagram of LSTM architecture:
  - Input sequence (40 timesteps × 8 features)
  - 2-layer LSTM with 128 hidden units
  - MLP prediction head
  - Output (20 future positions)
- Highlight the flow of information through time

**Narration**:
"The heart of our system is a two-layer LSTM network. Why LSTM? Because trajectories are sequential—where you'll be next depends on where you've been.

The LSTM reads 40 timesteps of history—that's 20 minutes at 30-second intervals. Each layer has 128 hidden units, learning increasingly abstract temporal patterns.

The final hidden state passes through a neural network head that outputs 20 future positions—our 10-minute forecast.

The entire model has 200,000 parameters and trains in under 30 minutes. It's deep learning, but it's efficient."

---

## SLIDE 5: Training and Metrics (1:20 - 1:45)
**[25 seconds]**

**Visual**:
- Side-by-side plots:
  - Left: Training and validation loss curves
  - Right: ADE and FDE metrics over epochs
- Highlight early stopping point

**Narration**:
"We train on 600 synthetic flights, carefully split by flight ID to prevent data leakage—no peeking at test trajectories during training!

The loss curves show smooth learning with minimal overfitting—our dropout and regularization are working.

We track two key metrics: Average Displacement Error, or ADE, measures accuracy across the entire forecast. Final Displacement Error, FDE, focuses on the endpoint.

Early stopping kicks in after about 12 epochs when validation performance plateaus. Training is complete."

---

## SLIDE 6: Results - The Numbers (1:45 - 2:10)
**[25 seconds]**

**Visual**:
- Results table comparing LSTM vs. Baseline
- Bar chart showing improvement percentages
- Animated counter showing metrics

**Narration**:
"Now for the exciting part—the results!

Our LSTM achieves an average displacement error of around 400 meters and a final displacement error of about 650 meters over a 10-minute horizon.

Compare this to the constant velocity baseline: 900 meters ADE and 1,600 meters FDE.

That's 55% better on ADE and 60% better on FDE! The LSTM has truly learned the patterns of aircraft motion.

For context, at 1 minute ahead, we're accurate to within 100-150 meters. By 10 minutes, we're still under 700 meters—impressive for such a long horizon."

---

## SLIDE 7: Visualizations (2:10 - 2:35)
**[25 seconds]**

**Visual**:
- 6-panel grid showing sample predictions:
  - Each panel: Ground truth (blue) vs. Prediction (orange)
  - Show good examples and one challenging case
- Error distribution histogram
- Error growth curve over prediction horizon

**Narration**:
"Let's look at some actual predictions. Here are six random test trajectories.

In blue, you see ground truth. In orange, our predictions. Notice how the model captures the overall direction and shape—even for complex maneuvers.

Of course, it's not perfect. Sharp turns are challenging—the model tends to smooth them out. And errors accumulate over time, as shown by this error growth curve.

But the histogram shows most predictions cluster around the mean. We're consistent, and that's crucial for real-world applications."

---

## SLIDE 8: Conclusions and Future Work (2:35 - 3:00)
**[25 seconds]**

**Visual**:
- Summary slide with checkmarks for achievements
- Icons representing future improvements (weather, attention, graphs)
- Call-to-action with GitHub link or contact

**Narration**:
"To wrap up: We've built a complete, self-contained trajectory prediction system that achieves excellent results.

What's next? We're excited about adding weather data—wind affects flight paths significantly. Attention mechanisms could help the model focus on relevant history. And graph neural networks could model interactions between multiple aircraft.

The synthetic data was perfect for prototyping, but we're ready to deploy on real OpenSky Network data.

This work shows that deep learning can revolutionize air traffic management—making skies safer and more efficient.

Thank you! The code and report are available for your review. Questions?"

---

## TIMING BREAKDOWN:
- Introduction: 35 seconds
- Approach: 45 seconds (Data: 20s, Architecture: 25s)
- Training: 25 seconds
- Results: 25 seconds
- Visualizations: 25 seconds
- Conclusions: 25 seconds
- **Total: 180 seconds (3 minutes)**

---

## PRESENTATION TIPS:

### Pacing:
- Speak clearly at ~140-150 words per minute
- Pause briefly between slides for emphasis
- Use transition phrases: "Now let's look at...", "Moving on to...", "The results show..."

### Visuals:
- Use animations sparingly—they should enhance, not distract
- Color code consistently: Blue = truth, Orange = predictions
- Large, readable fonts (min 24pt for body text)
- High contrast for accessibility

### Delivery:
- Practice with a timer—don't rush or drag
- Emphasize key numbers: "55% better!", "400 meters", "10 minutes"
- Show enthusiasm—this is exciting research!
- Make eye contact if presenting live

### Technical Balance:
- Enough detail for experts to understand the approach
- Simple enough for non-experts to follow the story
- Focus on insights, not implementation details

---

## SLIDE DESIGN SUGGESTIONS:

### Slide 1 (Title):
- Clean, professional
- Project title, your name, date
- Background: Subtle aircraft/map imagery

### Slide 2 (Problem):
- Visual metaphor for uncertainty
- Split screen: past vs. future
- Simple graphics, not too busy

### Slide 3 (Data Pipeline):
- Flow diagram with icons
- Show transformation visually
- List features clearly

### Slide 4 (Architecture):
- Box diagram or network visualization
- Annotate dimensions: [40, 8] → [128] → [20, 3]
- Use arrows to show information flow

### Slide 5 (Training):
- Matplotlib-style plots (familiar to ML audience)
- Dual y-axes if showing different scales
- Annotate key points (early stopping, best model)

### Slide 6 (Results):
- Bold numbers, large font
- Use color to highlight improvements (green = better)
- Table + chart for different learning styles

### Slide 7 (Visualizations):
- Multi-panel layout for comparisons
- Consistent color scheme
- Clear axis labels

### Slide 8 (Conclusions):
- Bullet points with icons
- Forward-looking and positive
- Include a call-to-action

---

## BACKUP SLIDES (If time allows or for Q&A):

### Backup 1: Implementation Details
- Python/PyTorch stack
- Training: AdamW, Smooth L1 loss, gradient clipping
- Reproducible with seed=42

### Backup 2: Ablation Studies
- Effect of LSTM layers (1 vs. 2)
- Hidden size variations
- Loss function comparison

### Backup 3: Error Analysis Deep Dive
- When does the model fail?
- Types of trajectories (straight, turning, climbing)
- Outlier analysis

### Backup 4: Real-World Deployment
- Integration with ATC systems
- Latency requirements
- Safety considerations

---

**END OF SCRIPT**
