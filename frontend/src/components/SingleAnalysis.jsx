import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  TextField,
  Typography,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Divider,
  CircularProgress,  
  Paper,
  Grid,
  Chip,
} from '@mui/material';
import { analyzeSingleTicket } from '../services/api';
import { useSnackbar } from 'notistack';

const SingleAnalysis = () => {
  const [ticket, setTicket] = useState({
    subject: '',
    description: '',
    priority: 'medium',
    category: '',
  });
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [quickMode, setQuickMode] = useState(false);
  const [errors, setErrors] = useState({});
  const { enqueueSnackbar } = useSnackbar();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setTicket((prev) => ({ ...prev, [name]: value }));
    // Clear error when user types
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const validate = () => {
    const newErrors = {};
    if (!ticket.subject.trim()) newErrors.subject = 'Subject is required';
    if (!ticket.description.trim()) newErrors.description = 'Description is required';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    setLoading(true);
  try {
    // Add timeout handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      enqueueSnackbar('Analysis is taking longer than expected...', { variant: 'info' });
    }, 45000); // Show warning after 45 seconds

    const result = await analyzeSingleTicket(
      ticket, 
      quickMode,
      { signal: controller.signal } // Pass abort signal
    );
    
    clearTimeout(timeoutId);
    setAnalysisResult(result);
    enqueueSnackbar('Analysis completed successfully!', { variant: 'success' });
  } catch (error) {
    if (error.name === 'AbortError') {
      enqueueSnackbar('Analysis is still processing. Please check back later.', { variant: 'warning' });
    } else {
      enqueueSnackbar(error.message, { variant: 'error' });
    }
  } finally {
    setLoading(false);
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
    switch (priority.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Single Ticket Analysis
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Ticket Details
              </Typography>
              
              <form onSubmit={handleSubmit}>
                <TextField
                  fullWidth
                  label="Subject"
                  name="subject"
                  value={ticket.subject}
                  onChange={handleChange}
                  margin="normal"
                  error={!!errors.subject}
                  helperText={errors.subject}
                  required
                />
                
                <TextField
                  fullWidth
                  label="Description"
                  name="description"
                  value={ticket.description}
                  onChange={handleChange}
                  margin="normal"
                  multiline
                  rows={4}
                  error={!!errors.description}
                  helperText={errors.description}
                  required
                />
                
                <TextField
                  fullWidth
                  label="Category (optional)"
                  name="category"
                  value={ticket.category}
                  onChange={handleChange}
                  margin="normal"
                />
                
                <FormControl component="fieldset" sx={{ mt: 2 }}>
                  <FormLabel component="legend">Priority</FormLabel>
                  <RadioGroup
                    row
                    name="priority"
                    value={ticket.priority}
                    onChange={handleChange}
                  >
                    <FormControlLabel value="low" control={<Radio />} label="Low" />
                    <FormControlLabel value="medium" control={<Radio />} label="Medium" />
                    <FormControlLabel value="high" control={<Radio />} label="High" />
                    <FormControlLabel value="critical" control={<Radio />} label="Critical" />
                  </RadioGroup>
                </FormControl>
                
                <FormControlLabel
                  control={
                    <RadioGroup
                      row
                      value={quickMode}
                      onChange={(e) => setQuickMode(e.target.value === 'true')}
                    >
                      <FormControlLabel value={false} control={<Radio />} label="Full Analysis" />
                      <FormControlLabel value={true} control={<Radio />} label="Quick Analysis" />
                    </RadioGroup>
                  }
                  label="Analysis Mode"
                  labelPlacement="start"
                  sx={{ mt: 2 }}
                />
                
                <Box sx={{ mt: 3 }}>
                  <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : null}
                  >
                    Analyze Ticket
                  </Button>
                </Box>
              </form>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          {analysisResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1">Ticket ID: {analysisResult.ticket_id}</Typography>
                </Box>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Category</Typography>
                    <Typography variant="body1">{analysisResult.category}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Priority</Typography>
                    <Chip
                      label={analysisResult.priority}
                      color={priorityColor(analysisResult.priority)}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Sentiment</Typography>
                    <Chip
                      label={analysisResult.sentiment.toFixed(2)}
                      color={sentimentColor(analysisResult.sentiment)}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary">Summary</Typography>
                    <Typography variant="body1">{analysisResult.summary}</Typography>
                  </Grid>
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle2" gutterBottom>
                  Suggested Response
                </Typography>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.100' }}>
                  <Typography variant="body2" whiteSpace="pre-wrap">
                    {analysisResult.suggested_response}
                  </Typography>
                </Paper>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle2" gutterBottom>
                  Analysis Details
                </Typography>
                <Typography variant="body2">
                  <strong>Word Count:</strong> {analysisResult.analysis_details.word_count}
                </Typography>
                <Typography variant="body2">
                  <strong>Analysis Method:</strong> {analysisResult.analysis_details.analysis_method}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default SingleAnalysis;