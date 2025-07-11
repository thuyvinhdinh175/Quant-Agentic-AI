import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, 
    Table, TableStyle, PageBreak, ListFlowable, ListItem
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import base64

class PDFReportGenerator:
    """
    A utility for generating PDF reports from financial analysis results.
    """
    
    def __init__(self, output_dir="data/reports"):
        """
        Initialize the PDF report generator.
        
        Args:
            output_dir: Directory to save the generated reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up styles
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='Title',
            fontName='Helvetica-Bold',
            fontSize=18,
            spaceAfter=12,
            alignment=1  # Center alignment
        ))
        self.styles.add(ParagraphStyle(
            name='Heading1',
            fontName='Helvetica-Bold',
            fontSize=16,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='Heading2',
            fontName='Helvetica-Bold',
            fontSize=14,
            spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            name='Normal',
            fontName='Helvetica',
            fontSize=10,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='Bullet',
            fontName='Helvetica',
            fontSize=10,
            leftIndent=20,
            spaceAfter=3
        ))
        self.styles.add(ParagraphStyle(
            name='PositiveHighlight',
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.green,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='NegativeHighlight',
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.red,
            spaceAfter=6
        ))
    
    def generate_stock_analysis_report(
        self, 
        symbol: str, 
        period: str,
        price_data: dict = None,
        technical_indicators: dict = None,
        risk_metrics: dict = None,
        sentiment_data: dict = None,
        figures: dict = None
    ):
        """
        Generate a comprehensive stock analysis PDF report.
        
        Args:
            symbol: Stock ticker symbol
            period: Analysis time period
            price_data: Dictionary containing price data
            technical_indicators: Dictionary containing technical indicator results
            risk_metrics: Dictionary containing risk analysis metrics
            sentiment_data: Dictionary containing sentiment analysis data
            figures: Dictionary containing matplotlib figure objects to include
            
        Returns:
            str: Path to the generated PDF report
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_Analysis_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create the document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=inch/2,
            leftMargin=inch/2,
            topMargin=inch/2,
            bottomMargin=inch/2
        )
        
        # Initialize story (content elements)
        story = []
        
        # Add title
        title = Paragraph(f"Financial Analysis Report: {symbol}", self.styles['Title'])
        story.append(title)
        
        # Add report metadata
        report_date = datetime.now().strftime("%B %d, %Y %H:%M")
        story.append(Paragraph(f"Generated on: {report_date}", self.styles['Normal']))
        story.append(Paragraph(f"Analysis Period: {period}", self.styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Add executive summary
        story.append(Paragraph("Executive Summary", self.styles['Heading1']))
        
        # Placeholder for summary - in real implementation, this would be generated dynamically
        executive_summary = """
        This report provides a comprehensive analysis of {symbol} stock over the {period} period. 
        The analysis includes price movements, technical indicators, risk assessment, and market sentiment.
        """.format(symbol=symbol, period=period)
        
        story.append(Paragraph(executive_summary, self.styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Add price analysis section
        if price_data:
            story.append(Paragraph("Price Analysis", self.styles['Heading1']))
            
            # Add price data summary
            story.append(Paragraph("Key Price Metrics", self.styles['Heading2']))
            
            price_metrics = [
                ["Metric", "Value"],
                ["Current Price", f"${price_data.get('current_price', 0):.2f}"],
                ["Period High", f"${price_data.get('period_high', 0):.2f}"],
                ["Period Low", f"${price_data.get('period_low', 0):.2f}"],
                ["Period Change", f"{price_data.get('period_change', 0):.2%}"],
                ["Average Volume", f"{price_data.get('avg_volume', 0):,.0f}"]
            ]
            
            price_table = Table(price_metrics, colWidths=[2*inch, 1.5*inch])
            price_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), colors.white),
                ('GRID', (0, 0), (1, -1), 1, colors.black)
            ]))
            
            story.append(price_table)
            story.append(Spacer(1, 0.25*inch))
            
            # Add price chart if available
            if figures and 'price_chart' in figures:
                img_data = self._figure_to_image(figures['price_chart'])
                img = Image(img_data, width=6*inch, height=3*inch)
                story.append(img)
            
            story.append(Spacer(1, 0.25*inch))
        
        # Add technical analysis section
        if technical_indicators:
            story.append(Paragraph("Technical Analysis", self.styles['Heading1']))
            
            for indicator_name, indicator_data in technical_indicators.items():
                story.append(Paragraph(f"{indicator_name} Analysis", self.styles['Heading2']))
                
                # Add indicator interpretation
                if 'interpretation' in indicator_data:
                    story.append(Paragraph(indicator_data['interpretation'], self.styles['Normal']))
                
                # Add indicator chart if available
                if figures and f'{indicator_name.lower()}_chart' in figures:
                    img_data = self._figure_to_image(figures[f'{indicator_name.lower()}_chart'])
                    img = Image(img_data, width=6*inch, height=3*inch)
                    story.append(img)
                
                story.append(Spacer(1, 0.25*inch))
            
            story.append(PageBreak())
        
        # Add risk analysis section
        if risk_metrics:
            story.append(Paragraph("Risk Analysis", self.styles['Heading1']))
            
            # Add risk metrics table
            story.append(Paragraph("Key Risk Metrics", self.styles['Heading2']))
            
            risk_metrics_table = [
                ["Metric", "Value"],
                ["Volatility (Ann.)", f"{risk_metrics.get('volatility', 0):.2%}"],
                ["Beta", f"{risk_metrics.get('beta', 0):.2f}"],
                ["Value at Risk (95%)", f"{risk_metrics.get('var_95', 0):.2%}"],
                ["Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}"],
                ["Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}"]
            ]
            
            risk_table = Table(risk_metrics_table, colWidths=[2*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), colors.white),
                ('GRID', (0, 0), (1, -1), 1, colors.black)
            ]))
            
            story.append(risk_table)
            story.append(Spacer(1, 0.25*inch))
            
            # Add risk assessment
            if 'risk_assessment' in risk_metrics:
                risk_style = 'NegativeHighlight' if risk_metrics['risk_assessment'].upper() == 'HIGH' else 'Normal'
                story.append(Paragraph(f"Risk Assessment: {risk_metrics['risk_assessment']}", self.styles[risk_style]))
            
            # Add risk charts if available
            if figures and 'risk_chart' in figures:
                img_data = self._figure_to_image(figures['risk_chart'])
                img = Image(img_data, width=6*inch, height=3*inch)
                story.append(img)
            
            story.append(Spacer(1, 0.25*inch))
        
        # Add sentiment analysis section
        if sentiment_data:
            story.append(Paragraph("Market Sentiment Analysis", self.styles['Heading1']))
            
            # Add sentiment summary
            avg_sentiment = sentiment_data.get('average_sentiment_score', 0)
            sentiment_trend = sentiment_data.get('sentiment_trend', 'Neutral')
            
            sentiment_style = 'Normal'
            if avg_sentiment > 0.2:
                sentiment_style = 'PositiveHighlight'
            elif avg_sentiment < -0.2:
                sentiment_style = 'NegativeHighlight'
            
            story.append(Paragraph(f"Average Sentiment Score: {avg_sentiment:.2f}", self.styles[sentiment_style]))
            story.append(Paragraph(f"Sentiment Trend: {sentiment_trend}", self.styles['Normal']))
            
            # Add sentiment distribution chart if available
            if figures and 'sentiment_chart' in figures:
                img_data = self._figure_to_image(figures['sentiment_chart'])
                img = Image(img_data, width=6*inch, height=3*inch)
                story.append(img)
            
            # Add top news
            if 'top_positive_news' in sentiment_data and sentiment_data['top_positive_news']:
                story.append(Paragraph("Top Positive News", self.styles['Heading2']))
                pos_news = sentiment_data['top_positive_news']
                
                bullet_list = []
                for news in pos_news:
                    bullet_text = f"{news.get('title', 'Unknown')} (Score: {news.get('sentiment_score', 0):.2f}, Source: {news.get('source', 'Unknown')})"
                    bullet_list.append(ListItem(Paragraph(bullet_text, self.styles['Bullet'])))
                
                story.append(ListFlowable(bullet_list, bulletType='bullet', start=None))
                story.append(Spacer(1, 0.15*inch))
            
            if 'top_negative_news' in sentiment_data and sentiment_data['top_negative_news']:
                story.append(Paragraph("Top Negative News", self.styles['Heading2']))
                neg_news = sentiment_data['top_negative_news']
                
                bullet_list = []
                for news in neg_news:
                    bullet_text = f"{news.get('title', 'Unknown')} (Score: {news.get('sentiment_score', 0):.2f}, Source: {news.get('source', 'Unknown')})"
                    bullet_list.append(ListItem(Paragraph(bullet_text, self.styles['Bullet'])))
                
                story.append(ListFlowable(bullet_list, bulletType='bullet', start=None))
                story.append(Spacer(1, 0.15*inch))
            
            # Add overall assessment
            if 'overall_assessment' in sentiment_data:
                story.append(Paragraph("Overall Assessment", self.styles['Heading2']))
                story.append(Paragraph(sentiment_data['overall_assessment'], self.styles['Normal']))
            
            story.append(Spacer(1, 0.25*inch))
        
        # Add conclusion
        story.append(Paragraph("Conclusion and Recommendations", self.styles['Heading1']))
        
        # In a real implementation, this would be dynamically generated based on all analysis
        conclusion = """
        Based on the comprehensive analysis of {symbol} over the {period} period, including price action,
        technical indicators, risk metrics, and market sentiment, the following conclusions can be drawn:
        
        [This section would include synthesized insights from all analysis sections.]
        """.format(symbol=symbol, period=period)
        
        story.append(Paragraph(conclusion, self.styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add disclaimer
        story.append(Paragraph("Disclaimer", self.styles['Heading2']))
        disclaimer = """
        This report is generated for informational purposes only and does not constitute investment advice.
        Past performance is not indicative of future results. Always conduct your own research before making
        investment decisions.
        """
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        # Build the PDF document
        doc.build(story)
        
        return filepath
    
    def _figure_to_image(self, fig):
        """
        Convert a matplotlib figure to an image that can be included in the PDF.
        
        Args:
            fig: matplotlib Figure object
            
        Returns:
            BytesIO: Image data
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf
