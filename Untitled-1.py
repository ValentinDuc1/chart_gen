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
    
    SUPPORTED_CHART_TYPES = ['line', 'bar', 'pie', 'scatter', 'horizontal_bar']
    
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
            chart_type: Type of chart ('line', 'bar', 'pie', 'scatter', 'horizontal_bar')
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
        if len(x) > 5:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
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