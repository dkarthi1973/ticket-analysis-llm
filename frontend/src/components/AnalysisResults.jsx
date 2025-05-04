import React from 'react';
import {
  Box,
  Typography,
  Divider,
  Paper,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';

const AnalysisResults = ({ results }) => {
  if (!results || results.length === 0) {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="body1" color="text.secondary">
          No analysis results to display.
        </Typography>
      </Box>
    );
  }

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

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        Detailed Analysis Results
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Ticket ID</TableCell>
                  <TableCell>Subject</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Priority</TableCell>
                  <TableCell>Sentiment</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((result, index) => (
                  <TableRow key={index}>
                    <TableCell>{result.ticket_id || `Ticket ${index + 1}`}</TableCell>
                    <TableCell>{result.summary || result.subject || '-'}</TableCell>
                    <TableCell>{result.category || '-'}</TableCell>
                    <TableCell>
                      <Chip
                        label={result.priority || 'medium'}
                        color={priorityColor(result.priority)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={result.sentiment?.toFixed(2) || '0.00'}
                        color={sentimentColor(result.sentiment || 0)}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>

        {results.map((result, index) => {
          // Get the analysis details safely
          const analysisDetails = result.analysis_details || {};
          
          return (
            <Grid item xs={12} key={index}>
              <Paper elevation={2} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  {result.ticket_id || `Ticket ${index + 1}`} - {result.summary || result.subject || 'No Subject'}
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">Category</Typography>
                    <Typography>{result.category || 'Uncategorized'}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">Priority</Typography>
                    <Chip
                      label={result.priority || 'medium'}
                      color={priorityColor(result.priority)}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">Sentiment</Typography>
                    <Chip
                      label={(result.sentiment || 0).toFixed(2)}
                      color={sentimentColor(result.sentiment || 0)}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">Word Count</Typography>
                    <Typography>{analysisDetails.word_count || 'N/A'}</Typography>
                  </Grid>
                </Grid>

                <Divider sx={{ my: 2 }} />

                {result.suggested_response && (
                  <>
                    <Typography variant="subtitle2" gutterBottom>
                      Suggested Response
                    </Typography>
                    <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100', mb: 2 }}>
                      <Typography variant="body2" whiteSpace="pre-wrap">
                        {result.suggested_response}
                      </Typography>
                    </Paper>
                  </>
                )}

                {(analysisDetails.analysis_method || analysisDetails.analysis_timestamp) && (
                  <>
                    <Typography variant="subtitle2" gutterBottom>
                      Analysis Details
                    </Typography>
                    
                    {analysisDetails.analysis_method && (
                      <Typography variant="body2">
                        <strong>Method:</strong> {analysisDetails.analysis_method}
                      </Typography>
                    )}
                    
                    {analysisDetails.analysis_timestamp && (
                      <Typography variant="body2">
                        <strong>Timestamp:</strong> {new Date(analysisDetails.analysis_timestamp).toLocaleString()}
                      </Typography>
                    )}
                  </>
                )}
              </Paper>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export default AnalysisResults;