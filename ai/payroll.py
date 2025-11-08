



from hmac import new
import json
import pandas as pd
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

importants= """
gross
net
charge
amount
Basic
Transport
Housing
NHF
NHIS
NSITF
tax
pension
employerPension
deduction
OtherAllowance
_id
fullname
id
employeeID.phone
employeeID.accountNumber
employeeID.pencomID
employeeID.annualSalary
employeeID.jobRole.name
employeeID.accountName
meta.annualGross
meta.sumBasicHousingTransport
meta.earnedIncome
meta.earnedIncomeAfterRelief
meta.sumRelief
"""

def important ():
    key_fields = [line.strip() for line in importants.strip().splitlines()]
    return key_fields


desired_columnsT2= [
     "type", "payment_status", "gross", "net", "charge", "amount", "_id", "Basic", "Housing", "Transport",
    "Leave", "Medical", "Bread", "Shoe", "Car", "House", "Nepa", "allOtherItems", "notPayableDetails",
    "Leave_schedule", "Medical_schedule", "Bread_schedule", "Shoe_schedule", "Car_schedule", "Nepa_schedule",
    "Basic_type", "Transport_type", "Housing_type", "Leave_type", "Medical_type", "Bread_type", "Shoe_type",
    "Car_type", "House_type", "Nepa_type", "reliefs", "NHF", "NHIS", "NSITF", "benefits", "benefit", "tax",
    "pension", "employerPension", "bonuses", "bonus", "deduction", "OtherAllowance", "allowance", "year",
    "paymentDate", "percentage", "month", "companyID", "CashAdvance", "leaveAllowance", "cashReimbursement",
    "__v", "createdAt", "updatedAt", "fullname", "id", "employeeID.companyLicense.assignedAt",
    "employeeID.companyLicense.licenseId", "employeeID.jobRole.jobRoleId", "employeeID.jobRole.name",
    "employeeID.employeeConfirmation.confirmed", "employeeID.employeeConfirmation.status",
    "employeeID.employeeConfirmation.date", "employeeID.termination.prorated.status",
    "employeeID.termination.status", "employeeID.termination.medicalBalance", "employeeID.termination.leaveBalance",
    "employeeID.termination.isPaid", "employeeID.termination.otherAllowancesBalance", "employeeID.termination.date",
    "employeeID.termination.reason", "employeeID.employeeType", "employeeID.employeeSubordinates",
    "employeeID.mentees", "employeeID.aboutEmployee", "employeeID.facebookUrl", "employeeID.twitterUrl",
    "employeeID.linkedInUrl", "employeeID.employeeTribe", "employeeID.allowanceType", "employeeID.annualSalary",
    "employeeID.costOfHire", "employeeID.hourlyRate", "employeeID.isActive", "employeeID.isConfirmed",
    "employeeID.dependents", "employeeID._id", "employeeID.employeeEmail", "employeeID.firstName",
    "employeeID.middleName", "employeeID.lastName", "employeeID.religion", "employeeID.gender", "employeeID.bankName",
    "employeeID.phone", "employeeID.accountNumber", "employeeID.salaryScheme._id", "employeeID.salaryScheme.name",
    "employeeID.salaryScheme.items", "employeeID.salaryScheme.country", "employeeID.salaryScheme.companyID",
    "employeeID.salaryScheme.employeeContribution", "employeeID.salaryScheme.employerContribution",
    "employeeID.salaryScheme.createdAt", "employeeID.salaryScheme.updatedAt", "employeeID.salaryScheme.__v",
    "employeeID.branchID._id", "employeeID.branchID.branchKeyName", "employeeID.branchID.branchName",
    "employeeID.branchID.companyID", "employeeID.branchID.createdOn", "employeeID.branchID.modifiedOn",
    "employeeID.branchID.createdAt", "employeeID.branchID.updatedAt", "employeeID.branchID.id",
    "employeeID.branchID.__v", "employeeID.staffID", "employeeID.employeeHireDate", "employeeID.dateOfBirth",
    "employeeID.employeeCadreStep", "employeeID.employeeCadre", "employeeID.createdOn", "employeeID.modifiedOn",
    "employeeID.companyID", "employeeID.companyFID", "employeeID.createdAt", "employeeID.updatedAt", "employeeID.id",
    "employeeID.__v", "employeeID.dailyPay", "employeeID.employeeManager", "employeeID.myx3ID", "employeeID.competency",
    "employeeID.departmentID", "employeeID.employementType", "employeeID.workArrangement", "employeeID.teamID",
    "employeeID.divisionID", "employeeID.accountName", "employeeID.bankCode", "employeeID.recipientCode",
    "employeeID.talentNominations", "employeeID.addonLicenses", "employeeID.leaveCategory", "employeeID.city",
    "employeeID.payRateType", "employeeID.businessUnitID", "employeeID.employeeCategory", "employeeID.lastHireDate",
    "employeeID.maritalStatus", "meta.annualGross", "meta.sumBasicHousingTransport", "meta.earnedIncome",
    "meta.earnedIncomeAfterRelief", "meta.sumRelief", "pfa", "taxAuthority", "employeeID.employeeConfirmation.processId",
    "employeeID.pencomID", "employeeID.profileImgUrl", "employeeID.nhfPIN", "employeeID.pfa", "employeeID.taxAuthority",
    "employeeID.taxID", "employeeID.promotionDate", "employeeID.bankCountry", "employeeID.branchCode",
    "employeeID.employeeTitle", "payslipPDFView.person.profileImgUrl", "payroll_id", "employee_ID", "Other Items"
]

desired_columnsT= [
     "type", "payment_status", "gross", "net", "charge", "amount", "_id", "Basic", "Housing", "Transport",
    "Leave", "Medical", "Bread", "Shoe", "Car", "House", "Nepa", "allOtherItems", "notPayableDetails",
    "Leave_schedule", "Medical_schedule", "Bread_schedule", "Shoe_schedule", "Car_schedule", "Nepa_schedule",
    "Basic_type", "Transport_type", "Housing_type", "Leave_type", "Medical_type", "Bread_type", "Shoe_type",
    "Car_type", "House_type", "Nepa_type", "reliefs", "NHF", "NHIS", "NSITF", "benefits", "benefit", "tax",
    "pension", "employerPension", "bonuses", "bonus", "deduction", "OtherAllowance", "allowance", "year",
    "paymentDate", "percentage", "month", "companyID", "CashAdvance", "leaveAllowance", "cashReimbursement",
    "__v", "createdAt", "updatedAt", "fullname", "id", "employeeID.companyLicense.assignedAt",
    "employeeID.companyLicense.licenseId", "employeeID.jobRole.jobRoleId", "employeeID.jobRole.name",
    "employeeID.employeeConfirmation.confirmed", "employeeID.employeeConfirmation.status",
    "employeeID.employeeConfirmation.date", "employeeID.termination.prorated.status",
    "employeeID.termination.status", "employeeID.termination.medicalBalance", "employeeID.termination.leaveBalance",
    "employeeID.termination.isPaid", "employeeID.termination.otherAllowancesBalance", "employeeID.termination.date",
    "employeeID.termination.reason", "employeeID.employeeType", "employeeID.employeeSubordinates",
    "employeeID.mentees", "employeeID.aboutEmployee", "employeeID.facebookUrl", "employeeID.twitterUrl",
    "employeeID.linkedInUrl", "employeeID.employeeTribe", "employeeID.allowanceType", "employeeID.annualSalary",
    "employeeID.costOfHire", "employeeID.hourlyRate", "employeeID.isActive", "employeeID.isConfirmed",
    "employeeID.dependents", "employeeID._id", "employeeID.employeeEmail", "employeeID.firstName",
    "employeeID.middleName", "employeeID.lastName", "employeeID.religion", "employeeID.gender", "employeeID.bankName",
    "employeeID.phone", "employeeID.accountNumber", "employeeID.salaryScheme._id", "employeeID.salaryScheme.name",
    "employeeID.salaryScheme.items", "employeeID.salaryScheme.country", "employeeID.salaryScheme.companyID",
    "employeeID.salaryScheme.employeeContribution", "employeeID.salaryScheme.employerContribution",
    "employeeID.salaryScheme.createdAt", "employeeID.salaryScheme.updatedAt", "employeeID.salaryScheme.__v",
    "employeeID.branchID._id", "employeeID.branchID.branchKeyName", 
    "employeeID.branchID.companyID", "employeeID.branchID.createdOn", "employeeID.branchID.modifiedOn",
    "employeeID.branchID.createdAt", 
    "employeeID.branchID.__v", "employeeID.staffID", "employeeID.employeeHireDate", "employeeID.dateOfBirth",
    "employeeID.employeeCadreStep", "employeeID.employeeCadre", "employeeID.createdOn",
    "employeeID.companyID", "employeeID.companyFID", "employeeID.createdAt", "employeeID.updatedAt", "employeeID.id",
    "employeeID.__v","employeeID.employeeManager", "employeeID.myx3ID", 
    "employeeID.departmentID", "employeeID.employementType", 
     "employeeID.city",
    "employeeID.payRateType", "employeeID.businessUnitID", "employeeID.employeeCategory", "employeeID.lastHireDate",
    "meta.annualGross", "meta.sumBasicHousingTransport",

]



def desire():
    desire = desired_columnsT
    return desire 

# Define your multiply function
def atb1(a, b):
    data1_raw = a
    data_raw = b
    systemprompt = f"""
You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets I will provide: a "previous period":{data1_raw} dataset and a "current period":{data_raw} dataset.
Your analysis must follow these steps:
Identify Employee Status: For every employee ID across both datasets, determine their status as one of the following:
Continuing: Appears in both datasets.
New: Appears only in the current period dataset.
Departed: Appears only in the previous period dataset.
Calculate Variances: Compute the monetary variance (NGN) for Gross Pay, Tax, and Pension for each employee and for the overall totals.
Identify Key Drivers: Analyze the variances to find the main reasons for the changes. Specifically look for:
Changes in headcount (new hires vs. departures).
Pay raises or decreases for continuing employees.
Unusual changes, such as a change in a deduction (like tax) without a corresponding change in gross pay. This is a critical insight to identify.
You must structure your output as a professional report using Markdown formatting with the following exact sections:
1. Executive Summary:
Start with the single most important number: the total variance in Gross Pay.
State whether this variance is favorable (a cost decrease) or unfavorable (a cost increase) from the company's perspective.
Briefly state the primary reason for this variance (e.g., "driven by headcount changes").
2. Overall Payroll Summary:
Create a summary table comparing the totals of the two periods.
The table columns must be: Metric, Previous Period, Current Period, Variance (NGN), and Variance (%).
Include rows for Gross Pay, Total Tax, and Total Pension.
3. Detailed Variance Analysis:
Create a sub-section titled 3.1. Headcount Changes that lists the departed and new employees and the gross pay impact of each group.
Create a sub-section titled 3.2. Variances for Continuing Employees that explicitly calls out any employees with changes in pay or deductions, specifying the exact variance amount.
4. Reconciliation of Gross Pay Variance:
Provide a simple table that clearly shows how the individual key drivers (e.g., Departures, New Hires, Pay Raises) sum up to the total Gross Pay variance. This proves your analysis is correct.
5. Conclusion & Recommendations:
Conclude with clear, actionable recommendations based on your findings. For example: "Verify the authorization for [Employee]'s pay raise" or "Investigate the reason for the tax change for [Employee], as their gross pay was unchanged."
Ensure the tone is professional, objective, and data-driven. Use currency formatting (e.g., N5,200) throughout the report."""
    responseY = llm.invoke([
        systemprompt,
        HumanMessage(content="Please review")
    ])
    return responseY.content



# retrieved_template1 = py.variance_prompt
# systemprompt=retrieved_template1

def atb(old, new,llmv,retrieved_template6):
    old = old
    new = new
    systemprompt = f"""

You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format:

"Previous Payroll Period": {old}

"Current Payroll Period": {new}

Your analysis must be meticulous, data-driven, and presented in a clear, professional Markdown report.

Analysis Instructions

You must follow these steps precisely:

1. Identify Employee Status:
Use employeeID._id as the unique identifier for each employee across both datasets. Classify each employee into one of the following categories:

Continuing: The employeeID._id exists in both the previous and current datasets.

New: The employeeID._id exists only in the current dataset.

Departed: The employeeID._id exists only in the previous dataset.

Suspicious: Flag any continuing employee as "Suspicious" if ANY of the following conditions are met. Comparison must be exact and case-sensitive.

fullname has changed.

employeeID.bankName has changed (e.g., "Zenith Bank" vs. "Zenith Banks").

employeeID.accountNumber has changed.

employeeID.phone has changed.

Suspicious (Duplicate ID): If an employeeID._id appears more than once within the payslips array of the Current Payroll Period, it must be flagged as a critical data integrity issue.

Note: If a file contains multiple top-level payroll objects, consolidate all payslips into a single list for each period before starting the analysis.

2. Calculate Monetary Variances:
For each employee and for the overall totals, compute the monetary variance (Current - Previous) in NGN for the following fields:

gross (Gross Pay)

tax (Tax)

pension (Employee Pension Contribution)

3. Identify Key Drivers of Variance:
Analyze the data to determine the root causes of any financial changes. Your analysis must explicitly connect variances to:

Headcount Changes: The financial impact of new hires and departures.

Pay Changes: Changes in gross pay for continuing employees.

Anomalies & Data Quality: The financial impact of suspicious records, especially duplicate entries.

Required Output Format (Markdown)

Generate the report using the exact structure and formatting below.

1. Executive Summary

Start with a headline figure: the total variance in Gross Pay.

State whether the variance is favorable (cost decrease) or unfavorable (cost increase).

Briefly summarize the primary drivers (e.g., headcount changes, significant pay adjustments, data anomalies).

2. Overall Payroll Summary

Provide a Markdown table comparing the aggregate values:

Generated markdown
| Metric        | Previous Period | Current Period | Variance (NGN) | Variance (%) |
|---------------|-----------------|----------------|----------------|--------------|
| Gross Pay     |                 |                |                |              |
| Total Tax     |                 |                |                |              |
| Total Pension |                 |                |                |              |

3. Detailed Variance Analysis
3.1 Headcount Changes

List new and departed employees and their financial impact.

Departed Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total Departures** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END

New Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total New Hires** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
3.2 Variances for Continuing & Suspicious Employees

Create a table for all continuing employees. Highlight significant changes and flag suspicious records.

Generated markdown
| Employee Name | Employee ID | Gross Pay Variance (NGN) | Notes & Flags |
|---------------|-------------|--------------------------|---------------|
| ...           |             |                          | 🔴 **Suspicious (Identity Change):** Bank name changed from 'Old Bank' to 'New Bank'. |
| ...           |             |                          | 🔴 **Suspicious (Duplicate ID):** Employee ID appears X times in the current period. |
| ...           |             |                          | **Significant Pay Change:** Describe the change (e.g., Housing increased by NXX). |
| ...           |             |                          | No significant variance. |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
4. Reconciliation of Gross Pay Variance

Summarize the drivers contributing to the total Gross Pay variance in a reconciliation table.

Generated markdown
| Driver                               | Count | Value Impact (NGN) |
|--------------------------------------|-------|--------------------|
| New Hires                            |       |                    |
| Departures                           |       |                    |
| Pay Changes (Continuing Employees)   |       |                    |
| Suspicious Anomalies (e.g., Duplicates) |       |                    |
| **Total Gross Pay Variance**         |       |                    |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
5. Conclusion & Recommendations

Provide clear, numbered, and actionable recommendations based on your findings. Prioritize critical issues.

🔴 URGENT: Investigate Duplicate Employee ID: Detail the specific employee and the risk of double payment.

🔴 URGENT: Verify Bank Detail Change: Detail the specific employee and the potential fraud risk.

Review Pay Increase Authorization: Specify the employee and the amount that needs verification.

Data Cleansing Protocol: Recommend a future action to prevent similar data integrity issues.

Final Instructions:

Format all monetary values with the Nigerian currency symbol and two decimal places (e.g., N5,200.00).

Maintain a professional, objective, and data-driven tone. Your primary goal is to act as a diligent analyst, highlighting not just the numbers but the underlying data quality issues and operational risks they represent.
"""

    systemprompt1= retrieved_template6

    responseY = llmv.invoke([
        systemprompt1,
        HumanMessage(content="Please review")
    ])
    return responseY.content

systempromptjjj = """

You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format:

"Previous Payroll Period": {old}

"Current Payroll Period": {new}

Your analysis must be meticulous, data-driven, and presented in a clear, professional Markdown report.

Analysis Instructions

You must follow these steps precisely:

1. Identify Employee Status:
Use employeeID._id as the unique identifier for each employee across both datasets. Classify each employee into one of the following categories:

Continuing: The employeeID._id exists in both the previous and current datasets.

New: The employeeID._id exists only in the current dataset.

Departed: The employeeID._id exists only in the previous dataset.

Suspicious: Flag any continuing employee as "Suspicious" if ANY of the following conditions are met. Comparison must be exact and case-sensitive.

fullname has changed.

employeeID.bankName has changed (e.g., "Zenith Bank" vs. "Zenith Banks").

employeeID.accountNumber has changed.

employeeID.phone has changed.

Suspicious (Duplicate ID): If an employeeID._id appears more than once within the payslips array of the Current Payroll Period, it must be flagged as a critical data integrity issue.

Note: If a file contains multiple top-level payroll objects, consolidate all payslips into a single list for each period before starting the analysis.

2. Calculate Monetary Variances:
For each employee and for the overall totals, compute the monetary variance (Current - Previous) in NGN for the following fields:

gross (Gross Pay)

tax (Tax)

pension (Employee Pension Contribution)

3. Identify Key Drivers of Variance:
Analyze the data to determine the root causes of any financial changes. Your analysis must explicitly connect variances to:

Headcount Changes: The financial impact of new hires and departures.

Pay Changes: Changes in gross pay for continuing employees.

Anomalies & Data Quality: The financial impact of suspicious records, especially duplicate entries.

Required Output Format (Markdown)

Generate the report using the exact structure and formatting below.

1. Executive Summary

Start with a headline figure: the total variance in Gross Pay.

State whether the variance is favorable (cost decrease) or unfavorable (cost increase).

Briefly summarize the primary drivers (e.g., headcount changes, significant pay adjustments, data anomalies).

2. Overall Payroll Summary

Provide a Markdown table comparing the aggregate values:

Generated markdown
| Metric        | Previous Period | Current Period | Variance (NGN) | Variance (%) |
|---------------|-----------------|----------------|----------------|--------------|
| Gross Pay     |                 |                |                |              |
| Total Tax     |                 |                |                |              |
| Total Pension |                 |                |                |              |

3. Detailed Variance Analysis
3.1 Headcount Changes

List new and departed employees and their financial impact.

Departed Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total Departures** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END

New Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total New Hires** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
3.2 Variances for Continuing & Suspicious Employees

Create a table for all continuing employees. Highlight significant changes and flag suspicious records.

Generated markdown
| Employee Name | Employee ID | Gross Pay Variance (NGN) | Notes & Flags |
|---------------|-------------|--------------------------|---------------|
| ...           |             |                          | 🔴 **Suspicious (Identity Change):** Bank name changed from 'Old Bank' to 'New Bank'. |
| ...           |             |                          | 🔴 **Suspicious (Duplicate ID):** Employee ID appears X times in the current period. |
| ...           |             |                          | **Significant Pay Change:** Describe the change (e.g., Housing increased by NXX). |
| ...           |             |                          | No significant variance. |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
4. Reconciliation of Gross Pay Variance

Summarize the drivers contributing to the total Gross Pay variance in a reconciliation table.

Generated markdown
| Driver                               | Count | Value Impact (NGN) |
|--------------------------------------|-------|--------------------|
| New Hires                            |       |                    |
| Departures                           |       |                    |
| Pay Changes (Continuing Employees)   |       |                    |
| Suspicious Anomalies (e.g., Duplicates) |       |                    |
| **Total Gross Pay Variance**         |       |                    |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
5. Conclusion & Recommendations

Provide clear, numbered, and actionable recommendations based on your findings. Prioritize critical issues.

🔴 URGENT: Investigate Duplicate Employee ID: Detail the specific employee and the risk of double payment.

🔴 URGENT: Verify Bank Detail Change: Detail the specific employee and the potential fraud risk.

Review Pay Increase Authorization: Specify the employee and the amount that needs verification.

Data Cleansing Protocol: Recommend a future action to prevent similar data integrity issues.

Final Instructions:

Format all monetary values with the Nigerian currency symbol and two decimal places (e.g., N5,200.00).

Maintain a professional, objective, and data-driven tone. Your primary goal is to act as a diligent analyst, highlighting not just the numbers but the underlying data quality issues and operational risks they represent.
"""



systemprompt11="""
 "Please perform a payroll variance analysis comparing two JSON files: {old} (the old payroll data) and {new} (the recent payroll data). The report should capture the following details:

New Employees: List all employees present in the new file but not in the old file, including their gross pay, net pay, and full account details (bank, account number, account name).

Salary Changes: Identify any employees present in both files whose gross, net, or charge amounts have changed. Specify the old and new values for each change.

Delisted Employees: List all employees present in the old file but not in the new file, including their last recorded gross pay, net pay, and full account details.

Changed in Account Details: For employees present in both files, identify any changes in their bank name, account number, or account name. Specify the old and new account details.

Any Other Significant Change: Provide a summary of the overall financial impact, including the total variance in gross payroll, net payroll, and charges between the old and new files.






"""

def systemprompt2(old, new):
    return f"""
You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format:

"Previous Payroll Period": {old}

"Current Payroll Period": {new}

Your analysis must be meticulous, data-driven, and returned as a structured JSON object.

🧾 Output Format Instructions

Return your analysis as a JSON object with the following top-level keys:

- "CautionInsights": [List of strings]
- "PredictiveInsights": [List of strings]
- "PositiveInsights": [List of strings]
- "PayrollComparisonSummary": {{
    "GrossPay": {{ "Previous": "", "Current": "", "Variance": "", "VariancePercent": "" }},
    "TotalTax": {{ "Previous": "", "Current": "", "Variance": "", "VariancePercent": "" }},
    "TotalPension": {{ "Previous": "", "Current": "", "Variance": "", "VariancePercent": "" }}
  }}
- "PayrollComparisonInsights": {{
    "NewEmployees": [ {{ "name": "", "id": "", "impact": "" }} ],
    "DepartedEmployees": [ {{ "name": "", "id": "", "impact": "" }} ]
  }}
- "ContinuingAndSuspiciousAnalysis": [ {{ "name": "", "id": "", "variance": "", "notes": "" }} ]
- "GrossPayReconciliation": {{
    "NewHires": {{ "count": 0, "impact": "" }},
    "Departures": {{ "count": 0, "impact": "" }},
    "PayChanges": {{ "count": 0, "impact": "" }},
    "SuspiciousAnomalies": {{ "count": 0, "impact": "" }},
    "TotalVariance": ""
  }}
- "Recommendations": [List of strings]

💡 Formatting Rules:
- Format all monetary values with the Nigerian currency symbol and two decimal places (e.g., ₦5,200.00).
- Format percentages with two decimal places (e.g., 6.25%).
- Maintain a professional, objective, and data-driven tone.
"""



def systemprompt(old, new):
    return f"""
You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format.

📊 Input Data
- "Previous Payroll Period": {old}
- "Current Payroll Period": {new}

🎯 Objective
Analyze the differences between the two payroll periods. Your analysis must be:
- Meticulous and data-driven
- Returned as a **single structured JSON object**
- **Do not return HTML, Markdown, or narrative text**
- **Do not include headings, tables, or formatting outside JSON**

🧾 Output Format Instructions
Return a JSON object with the following top-level keys and structure:

```json
{{
  "CautionInsights": ["..."],
  "PredictiveInsights": ["..."],
  "PositiveInsights": ["..."],
  "PayrollComparisonSummary": {{
    "GrossPay": {{ "Previous": "₦0.00", "Current": "₦0.00", "Variance": "₦0.00", "VariancePercent": "0.00%" }},
    "TotalTax": {{ "Previous": "₦0.00", "Current": "₦0.00", "Variance": "₦0.00", "VariancePercent": "0.00%" }},
    "TotalPension": {{ "Previous": "₦0.00", "Current": "₦0.00", "Variance": "₦0.00", "VariancePercent": "0.00%" }}
  }},
  "PayrollComparisonInsights": {{
    "NewEmployees": [{{ "name": "", "id": "", "impact": "₦0.00" }}],
    "DepartedEmployees": [{{ "name": "", "id": "", "impact": "₦0.00" }}]
  }},
  "ContinuingAndSuspiciousAnalysis": [{{ "name": "", "id": "", "variance": "₦0.00", "notes": "" }}],
  "GrossPayReconciliation": {{
    "NewHires": {{ "count": 0, "impact": "₦0.00" }},
    "Departures": {{ "count": 0, "impact": "₦0.00" }},
    "PayChanges": {{ "count": 0, "impact": "₦0.00" }},
    "SuspiciousAnomalies": {{ "count": 0, "impact": "₦0.00" }},
    "TotalVariance": "₦0.00"
  }},
  "Recommendations": ["..."]
}} """


def aluke():
#    return systemprompt
   return systemprompt



def get_payslips_from_json(json_file_path,desired_columns):
    # json_file_path is request.FILES.get('old') or request.FILES.get('new
    
    """
    Extracts payslips from a JSON file and returns a DataFrame with selected fields.
    
    Args:
        json_file_path (str): Path to the JSON file containing payroll data.
        
    Returns:
        pd.DataFrame: DataFrame containing the extracted payslips with selected fields.
    """
   
    data = json.load(json_file_path)
    if not isinstance(data, list):
     data = [data]
    #Notes
    all_payslips = []
    for payroll in data:
        payroll_id = payroll['_id']
        payslips = payroll.get('payslips', [])

        # Normalize payslips into a DataFrame
        df = pd.json_normalize(payslips)

        # Add payroll_id and payslip_id columns
        df['payroll_id'] = payroll_id
        df['employee_ID'] = df['_id']

        all_payslips.append(df)
    # Combine all into one DataFrame
    final_df = pd.concat(all_payslips, ignore_index=True)
    # desired_columns= desired_columns

    available_columns = [col for col in desired_columns if col in final_df.columns]
    # Trim safely using only available columns
    trimmed_df = final_df[available_columns]

    
    # Filter the final DataFrame
    # trimmed_df = final_df[desired_columns]
    json_output = trimmed_df.to_json(orient="records", indent=4)
    # # Save the final DataFrame to a CSV file
    # csv_file_path = r"C:\Users\Pro\Downloads\payslips_outputfullaboki.csv"
    # final_df.to_csv(csv_file_path, index=False)
        
    # df = pd.DataFrame(all_payslips)
    return json_output


def parse_markdown_to_json(markdown_text):
    # Your parsing logic goes here
    structured_data = {
        "CautionInsights": [],
        "PredictiveInsights": [],
        "PositiveInsights": [],
        "PayrollComparisonSummary": {},
        "PayrollComparisonInsights": {
            "NewEmployees": [],
            "DepartedEmployees": []
        },
        "ContinuingAndSuspiciousAnalysis": [],
        "GrossPayReconciliation": {},
        "Recommendations": []
    }

    # Example: Extract sections using headers or emojis
    # You would use regex or string splitting to populate structured_data

    return structured_data