from scipy.stats import chi2
# import math
# Chi-squared value
a = 79
b = 33
c = 165
d = 122


chi2_value = 86.67172

# Degrees of freedom
degrees_of_freedom = 1

# Calculate the p-value
p_value = chi2.sf(chi2_value, degrees_of_freedom)

level_of_signficance = 0.05

if p_value < level_of_signficance:
    print("The p-value is:{:.40f}".format(p_value), " Therefore signifies proof to reject the Hypothesis")
elif p_value > level_of_signficance:
    print("The p-value is:{:.40f}".format(p_value), " Therefore signifies proof to not reject the Hypothesis")

# print("p-value:", p_value)
# print("p-value: {:.40f}".format(p_value))
