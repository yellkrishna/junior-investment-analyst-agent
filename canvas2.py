from markdown_pdf import MarkdownPdf, Section

def markdown_to_pdf(markdown_text, output_file):
    # Initialize the MarkdownPdf object
    pdf = MarkdownPdf()

    # Add the Markdown content as a section
    pdf.add_section(Section(markdown_text))

    # Save the PDF to the specified file
    pdf.save(output_file)

# Example usage
final_report = """
# Financial Report

## Introduction

This is the introduction to the financial report.

## Analysis

Detailed analysis goes here.

## Conclusion

Final thoughts and conclusions.
"""

pdf_filename = 'Financial_Report.pdf'
markdown_to_pdf(final_report, pdf_filename)
print(f"The financial report has been converted to '{pdf_filename}'.")
