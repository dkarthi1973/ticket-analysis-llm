import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography, 
  CircularProgress,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  FormControlLabel,
  Switch,
  Alert, 
} from '@mui/material';
import { analyzeBulkTickets, getProcessingStatus } from '../services/api';
import { useSnackbar } from 'notistack';
import AnalysisResults from './AnalysisResults'; // Import the AnalysisResults component

const BulkAnalysis = () => {
  const [tickets, setTickets] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [quickMode, setQuickMode] = useState(true);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [error, setError] = useState(null);
  const [showDetailedResults, setShowDetailedResults] = useState(false); // State for toggling detailed view
  const { enqueueSnackbar } = useSnackbar();

  // Add debug logging
  console.log('BulkAnalysis component mounted');

  useEffect(() => {
    // Check if there are tickets from CSV upload in localStorage
    const storedTickets = localStorage.getItem('csvTickets');
    console.log('Checking localStorage for csvTickets:', storedTickets ? 'Found' : 'Not found');
    
    if (storedTickets) {
      try {
        const parsedTickets = JSON.parse(storedTickets);
        console.log('Parsed tickets:', parsedTickets);
        if (Array.isArray(parsedTickets) && parsedTickets.length > 0) {
          console.log('Setting tickets from localStorage, count:', parsedTickets.length);
          setTickets(parsedTickets);
          // Clear from localStorage to prevent reloading
          localStorage.removeItem('csvTickets');
          enqueueSnackbar(`Loaded ${parsedTickets.length} tickets from uploaded CSV`, { 
            variant: 'info',
            autoHideDuration: 3000
          });
        } else {
          console.log('Parsed tickets not valid, using sample data');
          loadSampleData();
        }
      } catch (e) {
        console.error('Error parsing stored tickets:', e);
        loadSampleData();
      }
    } else {
      console.log('No stored tickets, using sample data');
      loadSampleData();
    }
  }, [enqueueSnackbar]);

  const loadSampleData = () => {
    // Load sample data
    setTickets([
      {
        subject: 'Login issues',
        description: 'I cannot login to my account, getting error 500',
        priority: 'high',
      },
      {
        subject: 'Feature request',
        description: 'Please add dark mode to the application',
        priority: 'low',
      },
      {
        subject: 'Payment problem',
        description: 'I was charged twice for my subscription',
        priority: 'critical',
      },
    ]);
  };

  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(async () => {
        try {
          const status = await getProcessingStatus();
          setProcessingStatus(status);
          if (!status.is_processing) {
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error fetching status:', error);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleAnalyze = async () => {
    if (tickets.length === 0) {
      enqueueSnackbar('No tickets to analyze', { variant: 'warning' });
      return;
    }

    setLoading(true);
    setError(null);
    setShowDetailedResults(false); // Reset detailed view when starting new analysis
    setResults([]);
    setProcessingStatus({
      is_processing: true,
      total_tickets: tickets.length,
      processed_tickets: 0,
    });

    try {
      // Format tickets to match backend expectations
      const formattedTickets = tickets.map(ticket => ({
        subject: ticket.subject || '',
        description: ticket.description || '',
        priority: ticket.priority || 'medium',
        category: ticket.category || null, // Explicit null instead of undefined
        status: 'new',
        created_at: new Date().toISOString()
      }));
  
      console.log("Sending tickets to API:", formattedTickets);
      const analysisResults = await analyzeBulkTickets(formattedTickets, quickMode);
      console.log("Received results:", analysisResults);
      
      // If you inspect the structure of analysisResults here it will help debug
      console.log("Results structure:", 
        analysisResults.length > 0 ? Object.keys(analysisResults[0]) : "No results");
      
      setResults(analysisResults);
      
      // Show detailed results automatically when analysis is complete
      setShowDetailedResults(true);
      
      enqueueSnackbar(`Analyzed ${analysisResults.length} tickets successfully!`, { 
        variant: 'success',
        autoHideDuration: 3000
      });
    } catch (error) {
      console.error('Detailed error:', error);
      
      // Set the error state for display in the UI
      setError(error.message || 'Unknown error occurred');
      
      // Show error notification
      enqueueSnackbar(`Analysis failed: ${error.message || 'Unknown error'}`, { 
        variant: 'error',
        autoHideDuration: 5000
      });
    } finally {
      setLoading(false);
      setProcessingStatus(prev => ({
        ...prev,
        is_processing: false,
        estimated_completion: 'Complete'
      }));
    }
  };
  
  const sentimentColor = (score) => {
    if (score >= 0.5) return 'success';
    if (score >= 0.1) return 'info';
    if (score <= -0.5) return 'error';
    if (score <= -0.1) return 'warning';
    return 'default';
  };

  const priorityColor = (priority) => {
    if (!priority) return 'default';
    switch (priority.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  // Function to toggle detailed results view
  const toggleDetailedResults = () => {
    setShowDetailedResults(!showDetailedResults);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Bulk Ticket Analysis
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Ticket List ({tickets.length})
              </Typography>
              
              <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Subject</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Priority</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tickets.length > 0 ? (
                      tickets.map((ticket, index) => (
                        <TableRow key={index}>
                          <TableCell>{ticket.subject}</TableCell>
                          <TableCell>
                            {ticket.description?.length > 50 
                              ? `${ticket.description.substring(0, 50)}...` 
                              : ticket.description}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={ticket.priority || 'medium'}
                              color={priorityColor(ticket.priority)}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={3} align="center">
                          No tickets available. Upload a CSV or add tickets manually.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={quickMode}
                      onChange={(e) => setQuickMode(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Quick Analysis"
                />
                
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleAnalyze}
                  disabled={loading || tickets.length === 0}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                >
                  {loading ? 'Processing...' : 'Analyze All'}
                </Button>
              </Box>
              
              {processingStatus?.is_processing && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Processing {processingStatus.processed_tickets} of {processingStatus.total_tickets} tickets...
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {processingStatus.estimated_completion}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Results ({results.length})
              </Typography>
              
              {results.length > 0 ? (
                <>
                  <TableContainer component={Paper} sx={{ maxHeight: 500, overflow: 'auto' }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Subject</TableCell>
                          <TableCell>Category</TableCell>
                          <TableCell>Priority</TableCell>
                          <TableCell>Sentiment</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {results.map((result, index) => {
                          // Get matching original ticket if available, otherwise use result ticket ID
                          const matchingTicket = tickets.length > index ? tickets[index] : null;
                          
                          return (
                            <TableRow key={index}>
                              <TableCell>
                                {matchingTicket?.subject || result.summary || `Ticket ${result.ticket_id}`}
                              </TableCell>
                              <TableCell>{result.category}</TableCell>
                              <TableCell>
                                <Chip
                                  label={result.priority}
                                  color={priorityColor(result.priority)}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={result.sentiment.toFixed(2)}
                                  color={sentimentColor(result.sentiment)}
                                  size="small"
                                />
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button 
                      variant="outlined" 
                      color="primary"
                      onClick={toggleDetailedResults}
                    >
                      {showDetailedResults ? 'Hide Details' : 'Show Detailed Results'}
                    </Button>
                  </Box>
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No analysis results yet. Analyze tickets to see results.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Detailed Analysis Results Section */}
        {showDetailedResults && results.length > 0 && (
          <Grid item xs={12}>
            <AnalysisResults results={results} />
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default BulkAnalysis;