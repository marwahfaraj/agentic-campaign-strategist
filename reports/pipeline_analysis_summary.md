# Pipeline Analysis and Representative Outputs

This section demonstrates the end-to-end workflow: audience request, simulated SegmentAI audience creation, customer journey scoring, campaign-fit recommendation, and marketer guidance.

- **Input:** Create me a segment of game day heavy shoppers for a seasonal campaign  
  **Objective:** Promote seasonal demand  
  **Audience size:** 297 households  
  **Activation style:** Price-led coupon (86.5% confidence)  
  **Suggested message:** Limited-time savings: unlock a personalized offer on frozen.
- **Input:** Find lapsed snack buyers for a win-back coupon campaign  
  **Objective:** Win back lapsed shoppers  
  **Audience size:** 16 households  
  **Activation style:** Win-back reminder (32.6% confidence)  
  **Suggested message:** We saved something for you: fresh deals on snacks this week.
- **Input:** Audience of loyal dairy shoppers for a weekend bundle campaign  
  **Objective:** Grow basket size  
  **Audience size:** 297 households  
  **Activation style:** Price-led coupon (86.5% confidence)  
  **Suggested message:** Limited-time savings: unlock a personalized offer on frozen.
- **Input:** Create a segment of new shoppers for a personal care onboarding campaign  
  **Objective:** Increase repeat purchase  
  **Audience size:** 5 households  
  **Activation style:** Cross-sell bundle (35.3% confidence)  
  **Suggested message:** Complete the basket: pair beverages with a deal picked for your next trip.

## Output Artifacts

- `reports/tables/pipeline_demo_cases.csv`
- `reports/tables/pipeline_demo_cases.json`

## Discussion

These examples are useful for the final presentation because they show representative model inputs and outputs. In a production setting, the simulated audience creation layer would be replaced by an approved segmentation system and the weak labels would be replaced by validated campaign response or incrementality labels.
