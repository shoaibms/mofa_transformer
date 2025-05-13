"""
This script enhances an existing HTML Quality Control (QC) report by
embedding various visualizations. It parses the HTML content, injects
additional CSS for improved styling, and dynamically inserts plot images
(e.g., outlier summaries, SNR comparisons, spectral examples) into
relevant sections of the report. If plot images are not already present
in the specified directory, the script can generate them using the
SpectralPlotter class from the spectral_qc_plot module. The final output
is a new, enhanced HTML report.
"""
import os
import re
import sys

from bs4 import BeautifulSoup

# The sys.path.append is necessary for the local import of SpectralPlotter
# to work if the module is not in the Python path or installed.
# Per user request, this file path is not changed.
sys.path.append(r"C:\Users\ms\Desktop\hyper\augment\hyper")
from spectral_qc_plot import SpectralPlotter


def enhance_qc_report(html_path, output_path=None, qc_dir=None,
                      plots_dir=None, data_paths=None):
    """
    Enhance the existing QC HTML report by integrating visualizations.

    Parameters:
    -----------
    html_path : str
        Path to the original QC HTML report.
    output_path : str, optional
        Path to save the enhanced report (defaults to original name
        with "_enhanced" suffix).
    qc_dir : str, optional
        Directory containing QC results (defaults to the directory
        of html_path).
    plots_dir : str, optional
        Directory containing plot images (defaults to
        qc_dir/publication_plots).
    data_paths : dict, optional
        Dictionary with paths to original and augmented data, e.g.,
        {'original': '/path/to/original.csv',
         'augmented': '/path/to/augmented.csv'}.
    """
    if output_path is None:
        base, ext = os.path.splitext(html_path)
        output_path = f"{base}_enhanced{ext}"

    if qc_dir is None:
        qc_dir = os.path.dirname(html_path)

    if plots_dir is None:
        plots_dir = os.path.join(qc_dir, 'publication_plots')

    if not os.path.exists(plots_dir) or len(os.listdir(plots_dir)) == 0:
        print("Plots not found! Generating plots now...")

        plotter = SpectralPlotter(qc_dir, plots_dir)

        plotter.load_qc_data()
        plotter.plot_outlier_summary()
        plotter.plot_snr_comparison()
        plotter.plot_vegetation_indices()
        plotter.plot_range_check_heatmap()
        plotter.plot_overall_quality_radar()

        if data_paths and 'original' in data_paths and 'augmented' in data_paths:
            plotter.load_spectra_and_plot_examples(
                data_paths['original'], data_paths['augmented']
            )
    else:
        print(f"Using existing plots from: {plots_dir}")

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    style_tag = soup.find('style')
    additional_css = """
        .plot-container {
            margin: 20px 0;
            text-align: center;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-image {
            max-width: 100%;
            height: auto;
            margin: 10px auto;
        }
        .plot-caption {
            font-style: italic;
            color: #555;
            margin-top: 10px;
            text-align: center;
        }
        .section-divider {
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }
        .methods-section {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .two-column {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .column {
            flex: 1;
            min-width: 300px;
        }
        @media print {
            .plot-container {
                break-inside: avoid;
                page-break-inside: avoid;
            }
        }
    """
    if style_tag: # Ensure style_tag exists before trying to append to its string
        style_tag.string = (style_tag.string or "") + additional_css
    else: # If no style tag, create one and add it to head
        head_tag = soup.head
        if not head_tag: # If no head tag, create one (highly unlikely for valid HTML)
            head_tag = soup.new_tag('head')
            soup.html.insert(0, head_tag)
        new_style_tag = soup.new_tag('style')
        new_style_tag.string = additional_css
        head_tag.append(new_style_tag)


    plot_files = []
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]

    outlier_section = soup.find('h2', string=re.compile('Outlier Detection'))
    if outlier_section and 'outlier_summary.png' in plot_files:
        outlier_plot = create_plot_element(
            soup,
            'outlier_summary.png',
            'Outlier rates across different augmentation methods. Lower '
            'percentages indicate better statistical consistency.',
            plots_dir
        )
        outlier_section.find_next('table').insert_after(outlier_plot)

    snr_section = soup.find('h2', string=re.compile('Signal-to-Noise Assessment'))
    if snr_section and 'snr_comparison.png' in plot_files:
        snr_plot = create_plot_element(
            soup,
            'snr_comparison.png',
            'Signal-to-Noise Ratio comparison across augmentation methods. '
            'Higher values indicate better signal quality.',
            plots_dir
        )
        snr_section.find_next('table').insert_after(snr_plot)

    band_section = soup.find('h3', string=re.compile('Vegetation Indices'))
    if band_section and 'vegetation_indices.png' in plot_files:
        indices_plot = create_plot_element(
            soup,
            'vegetation_indices.png',
            'Preservation of key vegetation indices across augmentation '
            'methods. Values closer to zero indicate better preservation.',
            plots_dir
        )
        band_section.find_next('table').insert_after(indices_plot)

    range_section = soup.find('h2', string=re.compile('Range Checks'))
    if range_section and 'range_check_heatmap.png' in plot_files:
        range_plot = create_plot_element(
            soup,
            'range_check_heatmap.png',
            'Heatmap of physical constraint compliance. Darker green '
            'indicates higher compliance rates.',
            plots_dir
        )
        range_section.find_next('table').insert_after(range_plot)

    quality_section = soup.find('h2', string=re.compile('Overall Quality Assessment'))
    if quality_section and 'quality_radar.png' in plot_files:
        radar_plot = create_plot_element(
            soup,
            'quality_radar.png',
            'Radar chart showing quality metrics across different dimensions. '
            'Larger area indicates better overall quality.',
            plots_dir
        )
        # Ensure there is a next sibling to insert before, otherwise append
        next_sibling = quality_section.find_next_sibling()
        if next_sibling:
            next_sibling.insert_before(radar_plot)
        else:
            quality_section.parent.append(radar_plot)


    if 'spectral_examples.png' in plot_files and \
       'spectral_differences.png' in plot_files:
        spectral_section_header = soup.new_tag('h2')
        spectral_section_header.string = 'Spectral Data Visualization'

        spectral_desc = soup.new_tag('p')
        spectral_desc.string = (
            'Visual comparison of original and augmented spectral curves, '
            'demonstrating how different augmentation methods affect the '
            'spectral signatures.'
        )

        two_column_div = soup.new_tag('div', attrs={'class': 'two-column'})

        col1 = soup.new_tag('div', attrs={'class': 'column'})
        col1.append(create_plot_element(
            soup,
            'spectral_examples.png',
            'Comparison of original spectrum with examples from each '
            'augmentation method.',
            plots_dir
        ))

        col2 = soup.new_tag('div', attrs={'class': 'column'})
        col2.append(create_plot_element(
            soup,
            'spectral_differences.png',
            'Differences between augmented and original spectra, highlighting '
            'wavelength-specific effects.',
            plots_dir
        ))

        two_column_div.append(col1)
        two_column_div.append(col2)

        if quality_section:
            # Attempt to find the summary div associated with quality section
            summary_div = quality_section.find_next('div', class_='summary')
            insertion_point = summary_div if summary_div else quality_section

            divider = soup.new_tag('div', attrs={'class': 'section-divider'})
            insertion_point.insert_after(divider)
            divider.insert_after(spectral_section_header)
            spectral_section_header.insert_after(spectral_desc)
            spectral_desc.insert_after(two_column_div)
        else: # Fallback if quality_section is not found
            # Find a suitable place to insert, e.g., before the last major element or at the end of body
            body_tag = soup.body
            if body_tag:
                 # Insert before the last hr if it exists, otherwise append to body
                last_hr = body_tag.find_all('hr')
                insertion_point = last_hr[-1] if last_hr else body_tag
                
                divider = soup.new_tag('div', attrs={'class': 'section-divider'})
                if insertion_point is body_tag: # append if it's the body tag
                    insertion_point.append(divider)
                else: # insert_after if it's an element like hr
                    insertion_point.insert_after(divider)

                divider.insert_after(spectral_section_header)
                spectral_section_header.insert_after(spectral_desc)
                spectral_desc.insert_after(two_column_div)


    body_tag = soup.find('body')
    # Ensure body_tag exists and find_all 'hr' is not empty
    if body_tag:
        all_hr_tags = body_tag.find_all('hr')
        if all_hr_tags:
            last_hr_tag = all_hr_tags[-1]

            methods_section = soup.new_tag('div', attrs={'class': 'methods-section'})

            methods_title = soup.new_tag('h2')
            methods_title.string = 'Methodology'
            methods_section.append(methods_title)

            methods_desc = soup.new_tag('p')
            methods_desc.string = (
                'This report presents a comprehensive quality control analysis '
                'of spectral data augmentation. The analysis includes outlier '
                'detection using multiple statistical approaches, '
                'signal-to-noise ratio assessment, vegetation index '
                'preservation validation, and physical constraint verification.'
            )
            methods_section.append(methods_desc)

            methods_details = soup.new_tag('p')
            methods_details.string = (
                'The quality control pipeline implements a combined approach '
                'for outlier detection, using Z-score, IQR, Isolation Forest, '
                'and Local Outlier Factor methods. Signal quality is assessed '
                'through Savitzky-Golay filtering to separate signal from '
                'noise. Vegetation indices (NDVI, PRI, REIP) are calculated '
                'to verify preservation of biologically meaningful spectral '
                'features. Physical constraints verification ensures that '
                'augmented data adheres to fundamental properties of plant '
                'spectral signatures.'
            )
            methods_section.append(methods_details)

            citation_info = soup.new_tag('p')
            citation_info.string = (
                'Reference: Spectral Data Augmentation for Plant Stress '
                'Response Analysis, 2025.'
            )
            methods_section.append(citation_info)

            last_hr_tag.insert_before(methods_section)
        else: # If no <hr> tags, append methods section to the body
            # (Code to append methods_section to body_tag can be added here if needed)
            # For now, this case implies methods section might not be added as per original logic
            print("Warning: No <hr> tag found to insert Methodology section before.", file=sys.stderr)


    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify())) # Use prettify for better HTML output formatting

    print(f"\nEnhanced QC report generated: {output_path}")


def create_plot_element(soup, image_filename, caption_text, plots_dir):
    """
    Create a plot container element with image and caption.

    Parameters:
    -----------
    soup : BeautifulSoup
        BeautifulSoup object for HTML parsing.
    image_filename : str
        Filename of the plot image.
    caption_text : str
        Caption text for the plot.
    plots_dir : str
        Directory containing plot images.

    Returns:
    --------
    BeautifulSoup.Tag
        Plot container element.
    """
    rel_path = os.path.basename(plots_dir)

    container = soup.new_tag('div', attrs={'class': 'plot-container'})

    img = soup.new_tag('img', attrs={
        'src': f'{rel_path}/{image_filename}',
        'class': 'plot-image',
        'alt': caption_text
    })
    container.append(img)

    caption = soup.new_tag('div', attrs={'class': 'plot-caption'})
    caption.string = f'Figure: {caption_text}'
    container.append(caption)

    return container


if __name__ == "__main__":
    # Define file paths for the report enhancement
    # Note: The r"C:\Users\ms\..." paths are specific to an environment.
    # For broader usability, consider making these configurable or relative.
    qc_report_dir = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\quality_control"
    source_html_path = os.path.join(qc_report_dir, 'integrated_qc_report.html')
    enhanced_output_path = os.path.join(qc_report_dir, 'integrated_qc_report_enhanced.html')
    publication_plots_dir = os.path.join(qc_report_dir, 'publication_plots')

    # Define paths to data files used for generating spectral examples
    spectral_data_paths = {
        'original': r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv",
        'augmented': r"C:\Users\ms\Desktop\hyper\output\augment\augmented_spectral_data.csv"
    }

    # Call the main function to enhance the QC report
    enhance_qc_report(
        html_path=source_html_path,
        output_path=enhanced_output_path,
        qc_dir=qc_report_dir,
        plots_dir=publication_plots_dir,
        data_paths=spectral_data_paths
    )