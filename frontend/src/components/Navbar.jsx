import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <AppBar position="fixed">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Ticket Analysis System
        </Typography>
        <Box sx={{ display: 'flex', gap: '16px' }}>
          <Button color="inherit" component={Link} to="/">
            Dashboard
          </Button>
          <Button color="inherit" component={Link} to="/single">
            Single Analysis
          </Button>
          <Button color="inherit" component={Link} to="/bulk">
            Bulk Analysis
          </Button>
          <Button color="inherit" component={Link} to="/upload">
            Upload CSV
          </Button>
          <Button color="inherit" component={Link} to="/train">
            Train Classifier
          </Button>
          <Button color="inherit" component={Link} to="/model">
            Model Info
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;