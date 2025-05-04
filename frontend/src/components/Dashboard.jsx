import React, { useEffect, useState } from 'react';
import { Grid, Card, CardContent, Typography, CircularProgress } from '@mui/material';
import { getModelInfo } from '../services/api';
import ProcessingStatus from './ProcessingStatus';

const Dashboard = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [info] = await Promise.all([
          getModelInfo(),
        ]);
        setModelInfo(info);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '40px' }}>
        <CircularProgress />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ marginTop: '40px' }}>
        <Typography color="error">Error: {error}</Typography>
      </div>
    );
  }

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <ProcessingStatus />
      
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Information
              </Typography>
              <Typography variant="body1">
                <strong>Model Name:</strong> {modelInfo?.model_name}
              </Typography>
              <Typography variant="body1">
                <strong>Ollama URL:</strong> {modelInfo?.ollama_url}
              </Typography>
              <Typography variant="body1">
                <strong>Classifier Trained:</strong> {modelInfo?.classifier_trained ? 'Yes' : 'No'}
              </Typography>
              <Typography variant="body1">
                <strong>Categories:</strong> {modelInfo?.categories?.join(', ') || 'None'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Typography variant="body1" paragraph>
                - Analyze a single ticket
              </Typography>
              <Typography variant="body1" paragraph>
                - Upload and analyze a CSV file
              </Typography>
              <Typography variant="body1" paragraph>
                - Train the classifier with your data
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};

export default Dashboard;