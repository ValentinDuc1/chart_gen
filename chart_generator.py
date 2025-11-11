"""
Chart Generator for Automated Report Generation
Generates (chart PNG, JSON) pairs with matching filenames
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ChartGenerator:
    """
    Generate charts with corresponding JSON metadata files.
    Each chart is saved as a PNG with a matching JSON file.
    """
    
    SUPPORTED_CHART_TYPES = ['line', 'bar', 'pie', 'scatter', 'horizontal_bar', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution', 'hist2d']
    
    def __init__(self, output_dir: str = "./charts"):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory where chart files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        filename_root: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[str, str]:
        """
        Generate a chart and save both PNG and JSON files.
        
        Args:
            chart_type: Type of chart ('line', 'bar', 'pie', 'scatter', 'horizontal_bar', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution', 'hist2d')
            data: Chart data dictionary (format depends on chart_type)
            filename_root: Base filename without extension
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            metadata: Additional metadata to include in JSON
            **kwargs: Additional arguments passed to the chart function
            
        Returns:
            Tuple of (png_path, json_path)
        """
        if chart_type not in self.SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}. "
                           f"Supported types: {self.SUPPORTED_CHART_TYPES}")
        
        # Generate the chart
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        chart_method = getattr(self, f"_create_{chart_type}_chart")
        chart_method(ax, data, title, xlabel, ylabel, **kwargs)
        
        # Save PNG
        png_path = self.output_dir / f"{filename_root}.png"
        plt.tight_layout()
        # Add extra space at bottom for rotated labels
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(png_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)
        
        # Create and save JSON metadata
        json_data = {
            "filename_root": filename_root,
            "chart_type": chart_type,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "data": data,
            "generated_at": datetime.now().isoformat(),
            "png_file": f"{filename_root}.png",
            "metadata": metadata or {}
        }
        
        json_path = self.output_dir / f"{filename_root}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return str(png_path), str(json_path)
    
    def _create_line_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a line chart."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        if isinstance(y[0], list):
            # Multiple lines
            labels = data.get('labels', [f"Series {i+1}" for i in range(len(y))])
            for i, y_series in enumerate(y):
                ax.plot(x, y_series, marker='o', label=labels[i], 
                       linewidth=kwargs.get('linewidth', 2))
            ax.legend()
        else:
            # Single line
            ax.plot(x, y, marker='o', linewidth=kwargs.get('linewidth', 2))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _create_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a vertical bar chart."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        colors = kwargs.get('colors', None)
        ax.bar(x, y, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if they're strings and there are many
        if len(x) > 5 or any(len(str(label)) > 8 for label in x):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add extra bottom padding if labels are rotated
        if len(x) > 5:
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_horizontal_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a horizontal bar chart."""
        categories = data.get('categories', [])
        values = data.get('values', [])
        
        colors = kwargs.get('colors', None)
        ax.barh(categories, values, color=colors, alpha=0.8, 
                edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _create_discrete_distribution_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a discrete probability distribution chart as horizontal bars.
        Perfect for showing probability mass functions, frequency distributions,
        or categorical probability distributions.
        
        Data format:
        {
            'values': [0, 1, 2, 3, 4, 5],  # Discrete values (x-axis in vertical orientation)
            'probabilities': [0.05, 0.15, 0.25, 0.30, 0.20, 0.05],  # P(X=x)
            'labels': ['X=0', 'X=1', 'X=2', 'X=3', 'X=4', 'X=5']  # Optional custom labels
        }
        
        OR for named categories:
        {
            'categories': ['Excellent', 'Good', 'Fair', 'Poor'],
            'probabilities': [0.30, 0.40, 0.20, 0.10]
        }
        """
        import numpy as np
        
        # Check if we have categories or numeric values
        if 'categories' in data:
            categories = data['categories']
            labels = categories
        else:
            values = data.get('values', [])
            labels = data.get('labels', [f'X={v}' for v in values])
            categories = labels
        
        probabilities = data.get('probabilities', [])
        
        # Validate probabilities sum to approximately 1.0 (allow small rounding errors)
        prob_sum = sum(probabilities)
        if not (0.99 <= prob_sum <= 1.01):
            # Normalize if not already normalized
            probabilities = [p / prob_sum for p in probabilities]
        
        # Create color gradient based on probability values
        colors = kwargs.get('colors', None)
        if colors is None:
            # Create a color gradient from light to dark blue based on probability
            cmap = plt.cm.Blues
            norm = plt.Normalize(vmin=min(probabilities), vmax=max(probabilities))
            colors = [cmap(norm(p)) for p in probabilities]
        
        # Create horizontal bars
        y_positions = np.arange(len(categories))
        bars = ax.barh(y_positions, probabilities, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1)
        
        # Add probability labels on the bars
        show_values = kwargs.get('show_values', True)
        if show_values:
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                # Position label inside bar if it's wide enough, otherwise outside
                if width > 0.1:
                    ax.text(width - 0.02, bar.get_y() + bar.get_height()/2,
                           f'{prob:.3f}',
                           ha='right', va='center', fontsize=10, fontweight='bold',
                           color='white' if prob > 0.5 else 'black')
                else:
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{prob:.3f}',
                           ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Set labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel or 'Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or 'Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set x-axis limits to show full probability range
        ax.set_xlim(0, max(1.0, max(probabilities) * 1.1))
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add a vertical line at x=1.0 to show the full probability
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Total = 1.0')
        
        # Invert y-axis so highest value is at top (optional, controlled by kwargs)
        if kwargs.get('invert_y', False):
            ax.invert_yaxis()
        
        # Add mean/expected value line if requested
        if kwargs.get('show_expected_value', False) and 'values' in data:
            values = data['values']
            expected_value = sum(v * p for v, p in zip(values, probabilities))
            # Add as text annotation
            ax.text(0.98, 0.02, f'E[X] = {expected_value:.2f}',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
    
    def _create_pie_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a pie chart."""
        labels = data.get('labels', [])
        values = data.get('values', [])
        
        colors = kwargs.get('colors', None)
        explode = data.get('explode', None)
        
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors, explode=explode, textprops={'fontsize': 10})
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
    
    def _create_scatter_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a scatter plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        sizes = data.get('sizes', kwargs.get('marker_size', 50))
        colors = data.get('colors', kwargs.get('color', 'blue'))
        
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _create_grouped_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a grouped bar chart for multi-dimensional data.
        
        Data format:
        {
            'categories': ['Q1', 'Q2', 'Q3', 'Q4'],  # x-axis categories (e.g., time periods, regions)
            'groups': ['Product A', 'Product B', 'Product C'],  # different groups
            'values': [
                [100, 120, 115, 140],  # Product A values for each category
                [80, 95, 110, 105],     # Product B values
                [60, 70, 85, 95]        # Product C values
            ]
        }
        """
        import numpy as np
        
        categories = data.get('categories', [])
        groups = data.get('groups', [])
        values = data.get('values', [])
        
        x = np.arange(len(categories))
        width = 0.8 / len(groups)  # Width of bars
        
        colors = kwargs.get('colors', None)
        if not colors:
            # Default color palette
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        # Plot each group
        for i, (group_name, group_values) in enumerate(zip(groups, values)):
            offset = (i - len(groups)/2 + 0.5) * width
            ax.bar(x + offset, group_values, width, label=group_name,
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(categories) > 5 or any(len(str(cat)) > 8 for cat in categories):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_stacked_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a stacked bar chart for multi-dimensional data.
        
        Data format:
        {
            'categories': ['Q1', 'Q2', 'Q3', 'Q4'],  # x-axis categories (e.g., time periods, regions)
            'groups': ['Product A', 'Product B', 'Product C'],  # different groups
            'values': [
                [100, 120, 115, 140],  # Product A values for each category
                [80, 95, 110, 105],     # Product B values
                [60, 70, 85, 95]        # Product C values
            ]
        }
        """
        import numpy as np
        
        categories = data.get('categories', [])
        groups = data.get('groups', [])
        values = data.get('values', [])
        
        x = np.arange(len(categories))
        
        colors = kwargs.get('colors', None)
        if not colors:
            # Default color palette
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        # Plot each group as a stacked segment
        bottom = np.zeros(len(categories))
        for i, (group_name, group_values) in enumerate(zip(groups, values)):
            ax.bar(x, group_values, bottom=bottom, label=group_name,
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += np.array(group_values)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(categories) > 5 or any(len(str(cat)) > 8 for cat in categories):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_box_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a box plot (box and whisker chart) for distribution analysis.
        
        Data format:
        {
            'labels': ['Group A', 'Group B', 'Group C'],  # Category labels
            'data': [
                [10, 15, 13, 17, 20, 25, 18, 16],  # Group A data points
                [8, 12, 11, 14, 19, 16, 13, 15],    # Group B data points
                [12, 18, 15, 20, 22, 19, 17, 21]    # Group C data points
            ]
        }
        """
        labels = data.get('labels', [])
        values = data.get('data', [])
        
        # Create box plot
        box_parts = ax.boxplot(
            values,
            labels=labels,
            patch_artist=True,  # Fill boxes with color
            notch=kwargs.get('notch', False),  # Add notch for median confidence interval
            showmeans=kwargs.get('showmeans', True),  # Show mean as well as median
            meanline=kwargs.get('meanline', False)  # Show mean as line or point
        )
        
        # Color the boxes
        colors = kwargs.get('colors', None)
        if not colors:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style the other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element in box_parts:
                plt.setp(box_parts[element], color='black', linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(labels) > 5 or any(len(str(label)) > 8 for label in labels):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_area_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create an area chart using fill_between.
        
        Data format:
        {
            'x': [1, 2, 3, 4, 5],  # x-axis values
            'y': [10, 15, 13, 17, 20],  # Single area (fills from 0)
            
            # OR for range/band:
            'y1': [10, 12, 11, 14, 16],  # Lower bound
            'y2': [15, 18, 16, 20, 22]   # Upper bound
            
            # OR for multiple areas (stacked):
            'areas': [
                {'y': [10, 15, 13, 17, 20], 'label': 'Series 1'},
                {'y': [5, 8, 7, 9, 11], 'label': 'Series 2'}
            ]
        }
        """
        x = data.get('x', [])
        
        colors = kwargs.get('colors', None)
        if not colors:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        # Check which type of area chart
        if 'areas' in data:
            # Multiple stacked areas
            areas = data['areas']
            for i, area in enumerate(areas):
                y = area['y']
                label = area.get('label', f'Series {i+1}')
                ax.fill_between(x, y, alpha=0.7, color=colors[i % len(colors)], label=label)
                ax.plot(x, y, color=colors[i % len(colors)], linewidth=2)
            ax.legend(loc='best', fontsize=10)
            
        elif 'y1' in data and 'y2' in data:
            # Range/band chart (between y1 and y2)
            y1 = data['y1']
            y2 = data['y2']
            ax.fill_between(x, y1, y2, alpha=0.3, color=colors[0], label='Range')
            ax.plot(x, y1, color=colors[0], linewidth=2, linestyle='--', label='Lower Bound')
            ax.plot(x, y2, color=colors[1], linewidth=2, linestyle='--', label='Upper Bound')
            # Calculate and plot average
            avg = [(a + b) / 2 for a, b in zip(y1, y2)]
            ax.plot(x, avg, color='black', linewidth=2.5, label='Average')
            ax.legend(loc='best', fontsize=10)
            
        else:
            # Single area (from 0 to y)
            y = data.get('y', [])
            ax.fill_between(x, 0, y, alpha=0.6, color=colors[0])
            ax.plot(x, y, color=colors[0], linewidth=2.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Start y-axis at 0 for better visualization
        ax.set_ylim(bottom=0)
    
    def _create_hist2d_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a 2D histogram (heatmap) showing the density of points in 2D space.
        Perfect for visualizing the distribution and concentration of bivariate data.
        
        Data format:
        {
            'x': [1, 2, 3, 4, ...],  # x-coordinates of points
            'y': [5, 6, 7, 8, ...],  # y-coordinates of points
            'bins': [20, 20]  # Optional: number of bins for x and y (default: [20, 20])
        }
        """
        x = data.get('x', [])
        y = data.get('y', [])
        bins = data.get('bins', kwargs.get('bins', [20, 20]))
        
        # Get colormap
        cmap = kwargs.get('cmap', 'YlOrRd')  # Yellow-Orange-Red by default
        
        # Create 2D histogram
        hist, xedges, yedges, image = ax.hist2d(
            x, y, 
            bins=bins,
            cmap=cmap,
            alpha=0.9,
            edgecolors='none'
        )
        
        # Add colorbar to show density scale
        from matplotlib import pyplot as plt
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
        
        # Set labels and title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Add grid for better readability (behind the heatmap)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)
        
        # Add statistics as text annotation if requested
        if kwargs.get('show_stats', True):
            import numpy as np
            stats_text = f'n = {len(x)}\nμx = {np.mean(x):.1f}\nμy = {np.mean(y):.1f}'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                   fontsize=9, family='monospace')
    
    def _create_cohere_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a coherence plot showing the correlation between two signals
        as a function of frequency.
        
        Perfect for: Signal processing, correlation analysis, frequency-domain analysis
        
        Data format:
        {
            'x': [0.1, 0.2, 0.3, ...],      # First signal (time series)
            'y': [0.15, 0.25, 0.32, ...],   # Second signal (time series)
            'Fs': 1000,                      # Sampling frequency (optional, default: 1000)
            'NFFT': 256                      # FFT length (optional, default: 256)
        }
        """
        import numpy as np
        
        x = data.get('x', [])
        y = data.get('y', [])
        Fs = data.get('Fs', kwargs.get('Fs', 1000))  # Sampling frequency
        NFFT = data.get('NFFT', kwargs.get('NFFT', 256))  # FFT length
        
        # Compute coherence
        Cxy, freqs = ax.cohere(
            x, y,
            NFFT=NFFT,
            Fs=Fs,
            noverlap=NFFT // 2,  # 50% overlap
            scale_by_freq=True
        )
        
        # Enhance the plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel or 'Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or 'Coherence', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add horizontal line at coherence = 0.5 for reference
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label='Coherence = 0.5')
        
        # Add statistics if requested
        if kwargs.get('show_stats', True):
            mean_coherence = np.mean(Cxy)
            max_coherence = np.max(Cxy)
            max_freq = freqs[np.argmax(Cxy)]
            
            stats_text = (f'Mean: {mean_coherence:.3f}\n'
                        f'Max: {max_coherence:.3f}\n'
                        f'Peak: {max_freq:.1f} Hz')
            
            ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.8, edgecolor='gray'),
                fontsize=9, family='monospace')
        
        ax.legend(loc='lower left', fontsize=9)
    
    def batch_generate(
        self,
        chart_specs: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """
        Generate multiple charts at once.
        
        Args:
            chart_specs: List of chart specification dictionaries
            
        Returns:
            List of (png_path, json_path) tuples
        """
        results = []
        for spec in chart_specs:
            png_path, json_path = self.generate_chart(**spec)
            results.append((png_path, json_path))
        return results