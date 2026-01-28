# Methodology Notes

## Data Quality Decisions

### Smash Factor Limits (0.8 - 1.55)

The smash factor is the ratio of ball speed to club speed. For drivers:

- **USGA Limit**: The coefficient of restitution (COR) is limited to 0.830, which translates to a theoretical maximum smash factor of ~1.50
- **Upper bound (1.55)**: We use 1.55 to allow for minor measurement variance while catching clear errors
- **Lower bound (0.8)**: Below this indicates severe mishit (chunk, extreme heel/toe) or tracking error

### Spin Rate Bounds (0 - 10,000 rpm)

Driver spin rates vary significantly by player:
- **Tour average**: 2,400-2,700 rpm
- **Amateur average**: 3,000-3,500 rpm
- **Acceptable range**: 500-6,000 rpm for most normal shots
- **Upper bound**: 10,000 rpm catches sky balls (pop-ups) while allowing high-spin mishits

We observed one negative spin value (-21 billion) which was clearly a 32-bit integer overflow from Trackman firmware.

### Ball Speed Segmentation Thresholds

Segmentation by ball speed (not club speed) because:
1. Ball speed is the most reliably measured metric
2. It combines club speed AND strike quality
3. Trackman measures ball speed with higher precision

| Segment | Ball Speed | Club Speed (approx) | Typical Handicap |
|---------|-----------|---------------------|------------------|
| Elite | 145+ mph | 105+ mph | +2 to scratch |
| Scratch | 130-145 mph | 95-105 mph | 0-5 |
| Mid | 110-130 mph | 80-95 mph | 8-18 |
| High | <110 mph | <80 mph | 20+ |

---

## Statistical Methods

### ANOVA for Group Comparison

One-way ANOVA tests whether the means of multiple groups are equal. We use it to answer:
*"Does carry distance differ significantly across skill levels?"*

**Assumptions checked**:
- Independence of observations ✓ (different players)
- Approximate normality ✓ (large sample sizes)
- Homogeneity of variance ✓ (Levene's test)

**Effect Size (η²)**:
- Small: 0.01
- Medium: 0.06
- Large: 0.14

Our carry distance ANOVA shows η² = 0.70, meaning 70% of variance in carry is explained by skill level.

### Multiple Linear Regression

We predict carry distance from:
- Ball speed (mph)
- Launch angle (degrees)
- Spin rate (rpm)

**Model interpretation**:
```
carry = β₀ + β₁(ball_speed) + β₂(launch_angle) + β₃(spin_rate) + ε
```

Typical coefficients:
- β₁ ≈ 2.0: Each +1 mph ball speed adds ~2 yards
- β₂ ≈ 4.0-5.0: Each +1° launch adds ~4-5 yards (up to optimal)
- β₃ ≈ -0.003: Each +100 rpm costs ~0.3 yards

**R² Interpretation**:
- Our model achieves R² ≈ 0.61-0.65
- This is "good" but not "excellent" because:
  - Wind is not captured
  - Ground conditions vary
  - Individual swing characteristics matter

### Dispersion Ellipse Calculation

The 68% dispersion ellipse captures where most shots land:

1. Calculate covariance matrix of (carry, offline)
2. Eigendecomposition to get principal axes
3. Scale by chi-squared value for 68% confidence
4. Rotation angle from eigenvector direction

This is the same method used in Trackman's "Combine" app.

---

## Golf Domain Notes

### Why Higher Handicappers Show Tighter Dispersion

Counter-intuitive finding: High handicappers (±14 yds) show tighter dispersion than elite players (±25 yds).

**Explanation**: Offline distance is a function of:
1. Face angle at impact
2. Time of flight (longer = more curve)

Elite players hit it farther, so the ball has more time to curve offline. A 2° open face might curve 10 yards for an 85-yard shot but 25 yards for a 250-yard shot.

This insight is valuable for TaylorMade because it explains why forgiveness (MOI) matters differently across skill levels.

### Optimal Launch Conditions by Speed

Ball speed determines optimal launch/spin:

| Ball Speed | Optimal Launch | Optimal Spin | Expected Carry |
|------------|---------------|--------------|----------------|
| 165+ mph | 10-11° | 2,200-2,500 rpm | 285+ yds |
| 150-165 mph | 11-12° | 2,400-2,700 rpm | 250-285 yds |
| 135-150 mph | 12-13° | 2,700-3,000 rpm | 220-250 yds |
| <135 mph | 13-15° | 3,000-3,300 rpm | <220 yds |

Slower swing speeds need more launch and spin to maximize carry because they don't generate enough ball speed for a penetrating trajectory.

---

## Tools and Libraries

- **pandas**: Data manipulation and cleaning
- **numpy**: Numerical operations
- **scipy.stats**: ANOVA, regression, statistical tests
- **plotly**: Interactive visualizations
- **streamlit**: Dashboard deployment

---

## Future Enhancements

1. **K-means clustering**: Identify "good" vs "mishit" shot patterns
2. **Equipment A/B testing**: Framework for comparing drivers
3. **Session tracking**: Analyze improvement over time
4. **Iron data integration**: Extend beyond driver analysis
5. **Machine learning**: XGBoost for non-linear distance prediction

---

*Author: Eric Bolander*  
*Last Updated: January 2025*
